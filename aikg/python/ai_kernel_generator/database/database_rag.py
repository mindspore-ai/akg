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
import os
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ai_kernel_generator.database.vector_store import VectorStore
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.agent.utils.feature_extraction import FeatureExtraction
from ai_kernel_generator.utils.common_utils import get_md5_hash
from ai_kernel_generator.config.config_validator import load_config

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"

class DatabaseRAG(BaseRetriever):
    """算子优化方案RAG数据库系统"""
    # 必须显式声明所有字段
    database_path: str
    top_k: int
    vector_store: VectorStore

    def __init__(self, config_path: str, database_path: str = "", top_k: int = 5):
        """初始化RAG系统"""
        # 加载配置文件
        database_path = database_path if database_path else str(DEFAULT_DATABASE_PATH)
        top_k = top_k
        vector_store = VectorStore(config_path)

        super().__init__(
            database_path = database_path,
            top_k = top_k,
            vector_store=vector_store
        )

    def _get_relevant_documents(self, query: str, feature_invariants: str, *, run_manager=None):
        """实现基类要求的抽象方法"""
        return self.vector_store.vector_store.similarity_search(
            query=query,
            k=self.top_k,
            filter={"feature_invariants": feature_invariants}
        )

    def calculate_similarity(self, query: str, document: Document):
        """计算余弦相似度"""
        # 获取查询和文档的嵌入向量
        query_embed = self.vector_store.embedding_model.embed_query(query)
        doc_embed = self.vector_store.embedding_model.embed_query(document.page_content)

        # 计算余弦相似度
        dot_product = sum(q * d for q, d in zip(query_embed, doc_embed))
        norm_q = sum(q * q for q in query_embed) ** 0.5
        norm_d = sum(d * d for d in doc_embed) ** 0.5

        # 避免除以零
        return dot_product / (norm_q * norm_d + 1e-10)

    def feature_extractor(self, task_code: str, impl_type:str, backend:str, arch: str):
        """计算余弦相似度"""
        # 使用基类提供的标准方法检索文档
        # 特征提取
        feature_extractor = FeatureExtraction(
            task_desc=task_code,
            model_config=load_config().get("agent_model_config"),
            impl_type=impl_type,
            backend=backend,
            arch=arch
        )
        extracted_features, _, _ = asyncio.run(feature_extractor.run())
        return extracted_features.strip("```json").strip("```").strip()
    

    def sample(self, code: str, stragegy_mode: str = "random", backend: str = "", arch: str = "", impl_type: str = ""):
        """
        检索最相似的算子优化方案
        Args:
            operator_features (str): 算子特征描述
        Returns:
            list: 包含相似度、算子名称、文件路径和描述的字典列表
        """
        operator_features = self.feature_extractor(code, impl_type, backend, arch).replace("\n", ", ")
        feature_invariants = get_md5_hash(impl_type=impl_type, backend=backend, arch=arch)
        docs = self._get_relevant_documents(operator_features, feature_invariants)

        return [
            {
                "similarity_score": self.calculate_similarity(code, doc),
                "operator_name": doc.metadata["operator_name"],
                "operator_type": doc.metadata["operator_type"],
                "file_path": doc.metadata["file_path"],
                "description": doc.page_content
            }
            for doc in docs
        ]
    
    def insert(self, task_code, backend: str, arch: str, impl_type: str, framework:str = "", op_name: str = ""):
        """
        插入新的算子调度方案
        """
        framework_code = task_code.get("framework_code", "")
        impl_code = task_code.get("impl_code", "")

        md5_hash = get_md5_hash(impl_code=impl_code, op_name=op_name, impl_type=impl_type, backend=backend, arch=arch)

        operator_path = Path(self.database_path) / "operators"
        file_path = operator_path / arch / impl_type / md5_hash
        os.makedirs(file_path, exist_ok=True)
        if impl_code:
            impl_file = file_path / f"{impl_type}.py"
            with open(impl_file, "w", encoding="utf-8") as f:
                f.write(impl_code)
        
        if framework_code:
            if not framework:
                raise ValueError(f"framework={framework}：当提供框架代码时，必须指定框架名称")
            framework_file = file_path / f"{framework}.py"
            with open(framework_file, "w", encoding="utf-8") as f:
                f.write(framework_code)

        features = self.feature_extractor(task_code, impl_type, backend, arch)
        metadata_file = file_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
                f.write(features)

        self.vector_store.insert(arch, impl_type, md5_hash)

    def update(self):
        """
        更新已有算子调度方案
        """
        pass

    def delete(self, impl_code, backend: str, arch: str, impl_type: str, op_name: str = ""):
        """
        删除算子调度方案
        """
        md5_hash = get_md5_hash(impl_code=impl_code, op_name=op_name, impl_type=impl_type, backend=backend, arch=arch)

        operator_path = Path(self.database_path) / "operators"
        file_path = operator_path / arch / impl_type / md5_hash
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
