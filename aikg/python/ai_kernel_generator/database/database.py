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
import yaml
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

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"
DEFAULT_CONFIG_PATH = Path(get_project_root()) / "database" / "database_config.yaml"

class Database():
    def __init__(self, config_path: str = "", database_path: str = "", random_mode: bool = False):
        """初始化数据库系统"""
        self.database_path = database_path or str(DEFAULT_DATABASE_PATH)
        config_path = config_path or str(DEFAULT_CONFIG_PATH)
        if not random_mode:
            self.vector_store = VectorStore(config_path, self.database_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.random_mode = random_mode

    async def extract_features(self,impl_code: str, framework_code:str, backend:str, arch: str, dsl:str, profile=float('inf')):
        """提取任务特征"""
        # 特征提取
        feature_extractor = FeatureExtractor(
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
            "backend": backend,
            "arch": arch,
            "dsl": dsl,
            "description": parsed_content.description
        }
        return extracted_features
    
    def get_output_content(self, output_content:List[str], strategy_mode:RetrievalStrategy, docs, dsl, framework):
        result = []
        for doc in docs:
            case_path = Path(doc.metadata["file_path"])
            
            res_dict = {"strategy_mode": strategy_mode}
            for content in output_content:
                if content == "impl_code":
                    self._read_code_file(content, dsl, case_path, res_dict)
                    continue
                
                if content == "framework_code":
                    self._read_code_file(content, framework, case_path, res_dict)
                    continue
                
                metadata_file = case_path / "metadata.json"
                if not metadata_file.exists():
                    raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if content in metadata:
                    res_dict[content] = metadata[content]
                else:
                    raise ValueError(f"Content '{content}' not found in metadata. Available keys: strategy_mode, impl_code, framework_code, {', '.join(metadata.keys())}")

            result.append(res_dict)
        return result
    
    def randomicity_search(self, output_content: List[str], k: int = 5, backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        if not arch or not dsl:
            raise ValueError("arch and dsl must be provided in random mode")
            
        sample_path = Path(self.database_path) / "operators" / arch / dsl
        if not sample_path.exists() or not sample_path.is_dir():
            raise ValueError(f"Sample path {sample_path} does not exist or is not a directory")
        
        cases = [f for f in sample_path.iterdir() if f.is_dir()]
        k = min(k, len(cases))
        selected_cases = random.sample(cases, k) if k > 0 else []
        result = []
        for case_path in selected_cases:
            res_dict = {}
            for content in output_content:
                if content == "impl_code":
                    self._read_code_file(content, dsl, case_path, res_dict)
                elif content == "framework_code":
                    self._read_code_file(content, framework, case_path, res_dict)
                else:
                    raise ValueError(f"Invalid output content: {content}")
            result.append(res_dict)
        return result

    def max_marginal_relevance_search(self, query: str, feature_invariants: str, k: int = 5):
        return self.vector_store.loaded_vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k = max(20, 5 * k),
            lambda_mult = 0.2, # 极致多样性
            filter={"feature_invariants": feature_invariants}, 
        )
        
    def naivety_search(self, query: str, feature_invariants: str, k: int = 5):
        return self.vector_store.loaded_vectorstore.similarity_search(
            query=query,
            k=k,
            fetch_k = max(20, 5 * k),
            filter={"feature_invariants": feature_invariants}
        )
        
    def optimality_search(self, query: str, feature_invariants: str, k: int = 1):
        docs = self.vector_store.loaded_vectorstore.similarity_search(
            query=query,
            k= 5 * k,
            fetch_k = max(20, 10 * k),
            filter={"feature_invariants": feature_invariants}
        )

        min_profile = float('inf')
        min_doc = None
        for doc in docs:
            metadata_file = Path(doc.metadata["file_path"]) / "metadata.json"
            if not metadata_file.exists():
                continue
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            profile = metadata.get('profile', float('inf'))
            if profile < min_profile:
                min_profile = profile
                min_doc = doc
        
        return [min_doc]
    
    def rule_search(self, query: str, feature_invariants: str, k: int = 5, rule_desc:str = ""):
        return self.vector_store.loaded_vectorstore.similarity_search(
            query=query,
            k=k,
            fetch_k = max(20, 5 * k),
            filter={"feature_invariants": feature_invariants}
        )

    def samples_with_strategy(self, strategy_mode: RetrievalStrategy, output_content: List[str], features_str:str, feature_invariants:str,
                              sample_num: int = 5, rule_desc:str = "", backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        """根据指定的策略获取样本"""
        if strategy_mode == RetrievalStrategy.RANDOMICITY:
            result = self.randomicity_search(output_content, sample_num, backend, arch, dsl, framework)
        else:
            if strategy_mode == RetrievalStrategy.MMR:
                docs = self.max_marginal_relevance_search(features_str, feature_invariants, sample_num)
            elif strategy_mode == RetrievalStrategy.NAIVETY:
                docs = self.naivety_search(features_str, feature_invariants, sample_num)
            elif strategy_mode == RetrievalStrategy.OPTIMALITY:
                docs = self.optimality_search(features_str, feature_invariants, sample_num)
            elif strategy_mode == RetrievalStrategy.RULE:
                docs = self.rule_search(features_str, feature_invariants, sample_num, rule_desc)
            else:
                raise ValueError("Invalid strategy_mode")
            
            result = self.get_output_content(output_content, strategy_mode, docs, dsl, framework)
        return result

    async def samples(self, output_content: List[str], strategy_mode: RetrievalStrategy = RetrievalStrategy.NAIVETY, sample_num: int = 5, rule_desc: str = "",
                      impl_code: str = "", framework_code:str = "", backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        """
        基本采样，根据指定的策略获取样本
        """
        if not self.random_mode and strategy_mode != RetrievalStrategy.RANDOMICITY:
            features = await self.extract_features(impl_code, framework_code, backend, arch, dsl)
            features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
        else:
            features_str = ""
            strategy_mode = RetrievalStrategy.RANDOMICITY
        feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)
        result = self.samples_with_strategy(strategy_mode, output_content, features_str, feature_invariants, sample_num, rule_desc, backend, arch, dsl, framework)  
        return result
    
    async def combined_samples(self, strategy_mode: List[RetrievalStrategy], output_content: List[str], sample_num: List[int], rule_desc:str = "",
                               impl_code: str = "", framework_code:str = "",backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        """
        综合采样，根据不同的策略和数量获取样本
        """
        if len(strategy_mode) != len(sample_num):
            raise ValueError("strategy_mode and sample_num must have the same length")
        if not strategy_mode or not sample_num:
            raise ValueError("strategy_mode and sample_num cannot be empty")

        features = await self.extract_features(impl_code, framework_code, backend, arch, dsl)
        features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
        feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)
        result = []
        for strategy, num in zip(strategy_mode, sample_num):
            res = self.samples_with_strategy(strategy, output_content, features_str, feature_invariants, num, rule_desc, dsl, framework)
            result.extend(res)
        return result
    
    async def insert(self, impl_code:str, framework_code:str, backend: str, arch: str, dsl: str, framework: str, profile=float('inf')):
        """
        插入新的算子实现
        """
        md5_hash = get_md5_hash(impl_code=impl_code, backend=backend, arch=arch, dsl=dsl)
        operator_path = Path(self.database_path) / "operators"
        file_path = operator_path / arch / dsl / md5_hash

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

        if not self.random_mode:
            self.vector_store.insert(backend, arch, dsl, md5_hash)
        logger.info(f"Operator implementation inserted successfully, file path: {file_path}")

    def update(self):
        """
        更新已有算子实现
        """
        pass

    async def delete(self, impl_code:str, backend: str, arch: str, dsl: str):
        """
        删除算子实现
        """
        md5_hash = get_md5_hash(impl_code=impl_code, backend=backend, arch=arch, dsl=dsl)

        operator_path = Path(self.database_path) / "operators"
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
        if not self.random_mode:
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
    
    def clear(self):
        self.vector_store.clear()
        shutil.rmtree(str(Path(self.database_path) / "operators"))

    def _read_code_file(self, content: str, type_value: str, case_path: Path, res_dict: dict):
        """读取代码文件并添加到结果字典中"""
        if not type_value:
            return

        code_file_path = case_path / f"{type_value}.py"
        if not code_file_path.exists():
            raise FileNotFoundError(f"Code file not found: {code_file_path}")
        
        with open(code_file_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        res_dict[content] = code
