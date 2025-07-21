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
from pathlib import Path
import shutil
import yaml
import json
import logging
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.faiss import DistanceStrategy
from ai_kernel_generator.database.vector_store import VectorStore
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.agent.utils.feature_extraction import FeatureExtraction
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"
DEFAULT_CONFIG_PATH = Path(get_project_root()) / "database" / "rag_config.yaml"

class Database(BaseRetriever):
    """算子优化方案RAG数据库系统"""
    # 必须显式声明所有字段
    database_path: str
    vector_store: VectorStore
    config: dict

    def __init__(self, config_path: str = "", database_path: str = ""):
        """初始化RAG系统"""
        # 加载配置文件
        database_path = database_path or str(DEFAULT_DATABASE_PATH)
        config_path = config_path or str(DEFAULT_CONFIG_PATH)
        vector_store = VectorStore(config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        super().__init__(
            database_path=database_path,
            vector_store=vector_store,
            config=config
        )

    def _get_relevant_documents(self, query: str, feature_invariants: str, distance_strategy: str="COSINE", top_k: int = 5, *, run_manager=None):
        """实现基类要求的抽象方法"""
        if distance_strategy == "EUCLIDEAN_DISTANCE":
            strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
        elif distance_strategy == "MAX_INNER_PRODUCT":
            strategy = DistanceStrategy.MAX_INNER_PRODUCT
        elif distance_strategy == "DOT_PRODUCT":
            strategy = DistanceStrategy.DOT_PRODUCT
        elif distance_strategy == "JACCARD":
            strategy = DistanceStrategy.JACCARD
        elif distance_strategy == "COSINE":
            strategy = DistanceStrategy.COSINE
        else:
            raise ValueError(f"未知的距离策略: {distance_strategy}")

        return self.vector_store.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={"feature_invariants": feature_invariants}, 
            distance_strategy = strategy
        )

    def feature_extractor(self,impl_code: str, framework_code:str, impl_type:str, backend:str, arch: str):
        """提取任务特征"""
        # 特征提取
        feature_extractor = FeatureExtraction(
            model_config=self.config.get("agent_model_config"),
            impl_code=impl_code,
            framework_code=framework_code,
            impl_type=impl_type,
            backend=backend,
            arch=arch
        )
        feature_content, _, _ = asyncio.run(feature_extractor.run())
        parsed_content = feature_extractor.feature_parser.parse(feature_content)
        extracted_features = {
            "op_name": parsed_content.op_name,
            "op_type": parsed_content.op_type,
            "input_specs": parsed_content.input_specs,
            "output_specs": parsed_content.output_specs,
            "computation": parsed_content.computation,
            "schedule": parsed_content.schedule,
            "backend": parsed_content.backend,
            "arch": parsed_content.arch,
            "impl_type": parsed_content.impl_type,
            "description": parsed_content.description
        }
        return extracted_features
    

    def samples(self, impl_code: str, framework_code:str = "", top_k: int = 5, stragegy_mode: str = "random", backend: str = "", arch: str = "", impl_type: str = ""):
        """
        检索最相似的算子优化方案
        Returns:
            float：检索召回率
            list: 包含相似度、算子名称、文件路径和描述的字典列表
        """
        distance_strategy = self.config.get("distance_strategy") or "COSINE"
        verify_distance_strategy = self.config.get("verify_distance_strategy") or "EUCLIDEAN_DISTANCE"
        features = self.feature_extractor(impl_code, framework_code, impl_type, backend, arch)
        operator_features = ", ".join([f"{k}: {v}" for k, v in features.items()])
        feature_invariants = get_md5_hash(impl_type=impl_type, backend=backend, arch=arch)
        docs = self._get_relevant_documents(operator_features, feature_invariants, distance_strategy, top_k)

        recall = self.verify(operator_features, feature_invariants, docs, verify_distance_strategy, top_k)

        return recall, [
            {
                "similarity_score": score,
                "operator_name": doc.metadata["operator_name"],
                "file_path": doc.metadata["file_path"]
            }
            for doc, score in docs
        ]
    
    def insert(self, impl_code:str, framework_code:str, backend: str, arch: str, impl_type: str, framework: str):
        """
        插入新的算子实现
        """
        md5_hash = get_md5_hash(impl_code=impl_code, impl_type=impl_type, backend=backend, arch=arch)
        operator_path = Path(self.database_path) / "operators"
        file_path = operator_path / arch / impl_type / md5_hash
        
        features = self.feature_extractor(impl_code, framework_code, impl_type, backend, arch)
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

    def update(self):
        """
        更新已有算子实现
        """
        pass

    def delete(self, impl_code, backend: str, arch: str, impl_type: str):
        """
        删除算子实现
        """
        md5_hash = get_md5_hash(impl_code=impl_code, impl_type=impl_type, backend=backend, arch=arch)

        operator_path = Path(self.database_path) / "operators"
        file_path = operator_path / arch / impl_type / md5_hash
        if not file_path.exists():
            logger.warning(f"算子实现不存在：{file_path}")
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

    def verify(self, query, feature_invariants, sample_docs, distance_strategy="EUCLIDEAN_DISTANCE", top_k: int = 5):
        docs = self._get_relevant_documents(query, feature_invariants, distance_strategy, top_k)
        paths = [doc.metadata["file_path"] for doc, _ in docs]
        positives = len(sample_docs) 
        true_positives = 0
        for doc, _ in sample_docs:
            if doc.metadata["file_path"] in paths:
                true_positives += 1
        return true_positives / positives if positives > 0 else 0
