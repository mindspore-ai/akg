# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import shutil
import yaml
import json
import logging
from enum import Enum
from typing import List
from pathlib import Path
from ai_kernel_generator.database.vector_store import VectorStore
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.agent.utils.feature_extraction import FeatureExtraction
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

class RetrievalStragegy(Enum):
    RANDOMICITY = "randomicity"
    SIMILARITY = "similarity"
    OPTIMALITY = "optimality"
    RULE = "rule"

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"
DEFAULT_CONFIG_PATH = Path(get_project_root()) / "database" / "database_config.yaml"

class Database():
    def __init__(self, config_path: str = "", database_path: str = ""):
        """初始化数据库系统"""
        self.database_path = database_path or str(DEFAULT_DATABASE_PATH)
        config_path = config_path or str(DEFAULT_CONFIG_PATH)
        self.vector_store = VectorStore(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    async def feature_extractor(self,impl_code: str, framework_code:str, profile=float('inf')):
        """提取任务特征"""
        # 特征提取
        feature_extractor = FeatureExtraction(
            model_config=self.config.get("agent_model_config"),
            impl_code=impl_code,
            framework_code=framework_code
        )
        feature_content, _, _ = await feature_extractor.run()
        parsed_content = feature_extractor.feature_parser.parse(feature_content)
        extracted_features = {
            "op_name": parsed_content.op_name,
            "op_type": parsed_content.op_type,
            "input_specs": parsed_content.input_specs,
            "output_specs": parsed_content.output_specs,
            "computation": parsed_content.computation,
            "schedule": parsed_content.schedule,
            "profile": profile,
            "description": parsed_content.description
        }
        return extracted_features
    
    def get_output_content(self, output_content:List[str], stragegy_mode:RetrievalStragegy, docs, impl_type, framework):
        result = []
        for doc in docs:
            case_path = Path(doc.metadata["file_path"])
            
            res_dict = {"stragegy_mode": stragegy_mode}
            for content in output_content:
                if content == "impl_code" and impl_type:
                    code_file_path = case_path / f"{impl_type}.py"
                    with open(code_file_path, "r", encoding="utf-8") as f:
                        impl_code = f.read()
                    res_dict[content] = impl_code
                    continue
                
                if content == "framework_code" and framework:
                    code_file_path = case_path / f"{framework}.py"
                    with open(code_file_path, "r", encoding="utf-8") as f:
                        framework_code = f.read()
                    res_dict[content] = framework_code
                    continue
                
                metadata_file = case_path / "metadata.json"
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if content in metadata:
                    res_dict[content] = metadata[content]
                else:
                    raise ValueError(f"Content '{content}' not found in metadata. Available keys: {', '.join(metadata.keys())}")

            result.append(res_dict)
        return result
    
    def randomicity_search(self, query: str, feature_invariants: str, k: int = 5):
        return self.vector_store.loaded_vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k = max(20, 10 * k),
            lambda_mult = 0.2, # 极致多样性
            filter={"feature_invariants": feature_invariants}, 
        )
        
    def similarity_search(self, query: str, feature_invariants: str, k: int = 5):
        return self.vector_store.loaded_vectorstore.similarity_search(
            query=query,
            k=k,
            fetch_k = max(20, 10 * k),
            filter={"feature_invariants": feature_invariants}
        )
        
    def optimality_search(self, query: str, feature_invariants: str, k: int = 5):
        return self.vector_store.loaded_vectorstore.similarity_search(
            query=query,
            k=k,
            fetch_k = max(20, 10 * k),
            filter={"feature_invariants": feature_invariants}
        )

    def samples_with_stragegy(self, stragegy_mode: RetrievalStragegy, output_content: List[str], features_str:str, feature_invariants:str,
                              sample_num: int = 5, impl_type: str = "", framework: str = ""):
        """根据指定的策略获取样本"""
        if stragegy_mode == RetrievalStragegy.RANDOMICITY:
            docs = self.randomicity_search(features_str, feature_invariants, sample_num)
        elif stragegy_mode == RetrievalStragegy.SIMILARITY:
            docs = self.similarity_search(features_str, feature_invariants, sample_num)
        elif stragegy_mode == RetrievalStragegy.OPTIMALITY:
            docs = self.optimality_search(features_str, feature_invariants, sample_num)
        else:
            raise ValueError("Invalid stragegy_mode")
        
        result = self.get_output_content(output_content, stragegy_mode, docs, impl_type, framework)
        return result
    
    async def samples(self, output_content: List[str], stragegy_mode: RetrievalStragegy = RetrievalStragegy.SIMILARITY, sample_num: int = 5, rule_desc: str = "",
                impl_code: str = "", framework_code:str = "",backend: str = "", arch: str = "", impl_type: str = "", framework: str = ""):
        """
        基本采样，根据指定的策略获取样本
        """
        features = await self.feature_extractor(impl_code, framework_code)
        features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
        feature_invariants = get_md5_hash(impl_type=impl_type, backend=backend, arch=arch)
        
        result = self.samples_with_stragegy(stragegy_mode, output_content, features_str, feature_invariants, sample_num, impl_type, framework)
        return result
    
    async def combined_samples(self, stragegy_mode: List[RetrievalStragegy], output_content: List[str], sample_num: List[int], rule_desc:str = "",
                         impl_code: str = "", framework_code:str = "",backend: str = "", arch: str = "", impl_type: str = "", framework: str = ""):
        """
        综合采样，根据不同的策略和数量获取样本
        """
        if len(stragegy_mode) != len(sample_num):
            raise ValueError("stragegy_mode and sample_num must have the same length")
        if not stragegy_mode or not sample_num:
            raise ValueError("stragegy_mode and sample_num cannot be empty")

        features = await self.feature_extractor(impl_code, framework_code)
        features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
        feature_invariants = get_md5_hash(impl_type=impl_type, backend=backend, arch=arch)
        result = []
        for stragegy, num in zip(stragegy_mode, sample_num):
            res = self.samples_with_stragegy(stragegy, output_content, features_str, num, feature_invariants, impl_type, framework)
            result.extend(res)
        return result
    
    async def insert(self, impl_code:str, framework_code:str, backend: str, arch: str, impl_type: str, framework: str, profile=float('inf')):
        """
        插入新的算子实现
        """
        md5_hash = get_md5_hash(impl_code=impl_code, impl_type=impl_type, backend=backend, arch=arch)
        operator_path = Path(self.database_path) / "operators"
        file_path = operator_path / arch / impl_type / md5_hash
        
        features = await self.feature_extractor(impl_code, framework_code, profile)
        file_path.mkdir(parents=True, exist_ok=True)
        metadata_file = file_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=4)
        
        framework_file = file_path / f"{framework}.py"
        with open(framework_file, "w", encoding="utf-8") as f:
            f.write(framework_code)
        impl_file = file_path / f"{impl_type}.py"
        with open(impl_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        self.vector_store.insert(backend, arch, impl_type, md5_hash)
        logger.info(f"Operator implementation inserted successfully, file path: {file_path}")

    def update(self):
        """
        更新已有算子实现
        """
        pass

    async def delete(self, impl_code:str, backend: str, arch: str, impl_type: str):
        """
        删除算子实现
        """
        md5_hash = get_md5_hash(impl_code=impl_code, impl_type=impl_type, backend=backend, arch=arch)

        operator_path = Path(self.database_path) / "operators"
        file_path = operator_path / arch / impl_type / md5_hash
        if not file_path.exists():
            logger.warning(f"Operator implementation does not exist: {file_path}")
            return
        shutil.rmtree(file_path)
        # 删除空的上级目录
        current_dir = file_path.parent
        while current_dir.exists() and current_dir != operator_path:
            if not any(current_dir.iterdir()):
                current_dir.rmdir()
                current_dir = current_dir.parent
            else:
                break
        self.vector_store.delete(md5_hash)
        logger.info(f"Operator implementation deleted successfully, file path: {file_path}")

    def verify(self, query, feature_invariants, sample_docs, top_k: int = 5):
        docs = self.similarity_search(query, feature_invariants, top_k)
        paths = [doc.metadata["file_path"] for doc, _ in docs]
        positives = len(sample_docs) 
        true_positives = 0
        for doc, _ in sample_docs:
            if doc.metadata["file_path"] in paths:
                true_positives += 1
        return (true_positives / positives * 100) if positives > 0 else 0
