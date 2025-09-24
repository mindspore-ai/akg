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

import shutil
import json
import logging
import random
from enum import Enum
from typing import List
from pathlib import Path
from ai_kernel_generator.database.vector_store import VectorStore
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.agent.utils.feature_extractor import FeatureExtractor
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    RANDOMICITY = "randomicity"
    NAIVETY = "naivety"
    MMR = "max_marginal_relevance"
    OPTIMALITY = "optimality"
    RULE = "rule"
    HIERARCHY = "hierarchy"
    FUSION = "fusion"

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"

class Database():
    def __init__(self, database_path: str = "", vector_stores: List[VectorStore] = [], config: dict = None):
        """初始化数据库系统"""
        self.database_path = database_path or str(DEFAULT_DATABASE_PATH)
        self.vector_stores = vector_stores
        
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Database")

    async def extract_features(self,impl_code: str, framework_code:str, backend:str, arch: str, dsl:str, profile=float('inf')):
        """提取任务特征"""
        # 特征提取
        feature_extractor = FeatureExtractor(
            model_config=self.model_config,
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
            "backend": backend,
            "arch": arch,
            "dsl": dsl
        }
        return extracted_features

    def get_case_content(self, output_content: List[str], case_path: str, strategy_mode: RetrievalStrategy = None, dsl: str = "", framework: str = ""):
        case_path = Path(case_path)
        if not case_path.exists():
            raise FileNotFoundError(f"Case path not found: {case_path}")
        
        res_dict = {"strategy_mode": strategy_mode}
        for content in output_content:
            if content == "impl_code" and dsl:
                    code_file_path = case_path / f"{dsl}.py"
                    if not code_file_path.exists():
                        raise FileNotFoundError(f"Code file not found: {code_file_path}")
                    with open(code_file_path, "r", encoding="utf-8") as f:
                        impl_code = f.read()
                    res_dict[content] = impl_code
                    continue
                
            if content == "framework_code" and framework:
                code_file_path = case_path / f"{framework}.py"
                if not code_file_path.exists():
                    raise FileNotFoundError(f"Code file not found: {code_file_path}")
                with open(code_file_path, "r", encoding="utf-8") as f:
                    framework_code = f.read()
                res_dict[content] = framework_code
                continue
            
            metadata_file = case_path / "metadata.json"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if content in metadata:
                res_dict[content] = metadata[content]
            else:
                raise ValueError(f"Content '{content}' not found. Available keys: strategy_mode, impl_code, framework_code, {', '.join(metadata.keys())}")

        return res_dict

    def randomicity_search(self, output_content: List[str], k: int = 5, backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        if not arch or not dsl:
            raise ValueError("arch and dsl must be provided in random mode")
            
        sample_path = Path(self.database_path) / arch / dsl
        if not sample_path.exists() or not sample_path.is_dir():
            raise ValueError(f"Sample path {sample_path} does not exist or is not a directory")
        
        cases = [f for f in sample_path.iterdir() if f.is_dir()]
        k = min(k, len(cases))
        selected_cases = random.sample(cases, k) if k > 0 else []
        result = []
        for case_path in selected_cases:
            result.append(self.get_case_content(output_content, case_path, RetrievalStrategy.RANDOMICITY, dsl, framework))
        return result
    
    def get_output_content(self, output_content:List[str], strategy_mode:RetrievalStrategy, docs, dsl, framework):
        result = []
        for doc in docs:
            case_path = doc.metadata["file_path"]
            result.append(self.get_case_content(output_content, case_path, strategy_mode, dsl, framework))
        return result
    
    async def samples(self, output_content: List[str], strategy_modes: List[RetrievalStrategy] = [], sample_num: int = 5, rule_desc: str = "",
                      impl_code: str = "", framework_code:str = "", backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        """
        基本采样，根据指定的策略获取样本
        """        
        if len(strategy_modes) != len(self.vector_stores):
            raise ValueError("strategy_modes和vector_stores长度必须一致")
        
        if not self.vector_stores:
            return self.randomicity_search(output_content, sample_num, backend, arch, dsl, framework)
        
        need_extract_features = False
        for vector_store, strategy_mode in zip(self.vector_stores, strategy_modes):
            if vector_store.enable_vector_store and strategy_mode!= RetrievalStrategy.RANDOMICITY:
                need_extract_features = True
                break

        if need_extract_features:
            features = await self.extract_features(impl_code, framework_code, backend, arch, dsl)
            features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
            feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)

        for vector_store, strategy_mode in zip(self.vector_stores, strategy_modes):
            if vector_store.enable_vector_store and strategy_mode != RetrievalStrategy.RANDOMICITY:
                if strategy_mode == RetrievalStrategy.MMR:
                    docs = vector_store.max_marginal_relevance_search(features_str, feature_invariants, sample_num)
                elif strategy_mode == RetrievalStrategy.NAIVETY:
                    docs = vector_store.similarity_search(features_str, feature_invariants, sample_num)
                else:
                    raise ValueError("Invalid strategy_mode")
                result = self.get_output_content(output_content, strategy_mode, docs, dsl, framework)
            else:
                result = self.randomicity_search(output_content, sample_num, backend, arch, dsl, framework)
            return result
    
    async def insert(self, impl_code:str, framework_code:str, backend: str, arch: str, dsl: str, framework: str, profile=float('inf')):
        """
        插入新的算子实现
        """
        md5_hash = get_md5_hash(impl_code=impl_code, backend=backend, arch=arch, dsl=dsl)
        file_path = Path(self.database_path) / arch / dsl / md5_hash

        features = await self.extract_features(impl_code, framework_code, backend, arch, dsl, profile)
        file_path.mkdir(parents=True, exist_ok=True)
        metadata_file = file_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=4)
        
        framework_file = file_path / f"{framework}.py"
        with open(framework_file, "w", encoding="utf-8") as f:
            f.write(framework_code)
        impl_file = file_path / f"{dsl}.py"
        with open(impl_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        for vector_store in self.vector_stores:
            vector_store.insert(f"{arch}/{dsl}/{md5_hash}")
        logger.info(f"Operator implementation inserted successfully, file path: {file_path}")

    async def delete(self, impl_code:str, backend: str, arch: str, dsl: str):
        """
        删除算子实现
        """
        md5_hash = get_md5_hash(impl_code=impl_code, backend=backend, arch=arch, dsl=dsl)

        operator_path = Path(self.database_path)
        file_path = operator_path / arch / dsl / md5_hash
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

        for vector_store in self.vector_stores:
            vector_store.delete(f"{arch}/{dsl}/{md5_hash}")
        logger.info(f"Operator implementation deleted successfully, file path: {file_path}")
    
    def clear(self):
        for vector_store in self.vector_stores:
            vector_store.clear()
        shutil.rmtree(self.database_path)

