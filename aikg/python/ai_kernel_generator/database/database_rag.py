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
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ai_kernel_generator.database.vector_store import VectorStore
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.agent.utils.feature_extraction import FeatureExtraction
from ai_kernel_generator.utils.common_utils import get_md5_hash

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
            task_code=task_code,
            model_config={"feature_extraction": self.model_config},
            impl_type=impl_type,
            backend=backend,
            arch=arch
        )
        extracted_features, _, _ = asyncio.run(feature_extractor.run())
        return ', '.join([f"{k}: {v}" for k, v in extracted_features.items()])
    

    def sample(self, code: str, stragegy_mode: str = "random", backend: str = "", arch: str = "", impl_type: str = ""):
        """
        检索最相似的算子优化方案
        Args:
            operator_features (str): 算子特征描述
        Returns:
            list: 包含相似度、算子名称、文件路径和描述的字典列表
        """
        operator_features = self.feature_extractor(code, impl_type, backend, arch)
        feature_invariants = get_md5_hash(impl_type, backend, arch)
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
    
    def insert(self, task_code, op_name: str = "", framework:str = "", backend: str = "", arch: str = "", impl_type: str = ""):
        """
        插入新的算子调度方案
        """
        md5_hash = get_md5_hash(task_code, op_name=op_name, impl_type=impl_type, backend=backend, arch=arch)

        framework_code = task_code.get("framework_code", "")
        designer_code = task_code.get("designer_code", "")
        coder_code = task_code.get("coder_code", "")
        operator_path = Path(self.database_path) / "operators"
        impl_file = operator_path / arch / op_name / md5_hash
        if designer_code:
            impl_file = impl_file / "aul.py"
            with open(impl_file, "w", encoding="utf-8") as f:
                f.write(designer_code)
        
        if coder_code:
            impl_file = impl_file / f"{impl_type}.py"
            with open(impl_file, "w", encoding="utf-8") as f:
                f.write(task_code)
        
        if framework_code:
            impl_file = impl_file / f"{framework}.py"
            with open(impl_file, "w", encoding="utf-8") as f:
                f.write(framework_code)

        self.vector_store.insert(op_name, arch)

    def update(self):
        """
        更新已有算子调度方案
        """
        pass

    def delete(self, op_name, arch, type=""):
        """
        删除算子调度方案
        """
        operator_path = Path(self.database_path) / "operators"
        file_path = operator_path / arch / op_name
        if type:
            file_path = file_path / f"{type}.py"
            file_path.unlink()
        else:
            shutil.rmtree(file_path)
        self.vector_store.delete(op_name, arch)
    
    def test_insert(self, op_name, arch):
        self.vector_store.insert(op_name, arch)
