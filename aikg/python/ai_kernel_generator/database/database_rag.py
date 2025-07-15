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

from pathlib import Path
# from langchain_core.documents import Document
# from langchain_core.retrievers import BaseRetriever
from ai_kernel_generator.database.vector_store import VectorStore
from ai_kernel_generator.core.utils import ParsedCode
from ai_kernel_generator import get_project_root

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

    def _get_relevant_documents(self, query: str, operator_features: str, *, run_manager=None):
        """实现基类要求的抽象方法"""
        tmp = self.vector_store._load_or_create_vector_store()
        return tmp.vector_store.similarity_search(
            query=query,
            k=self.top_k,
            filter={"arch": operator_features["arch"]}
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
    
    def find(self, operator_features: str):
        """
        检索最相似的算子优化方案
        Args:
            operator_features (str): 算子特征描述
        Returns:
            list: 包含相似度、算子名称、文件路径和描述的字典列表
        """
        # 使用基类提供的标准方法检索文档
        docs = self.get_relevant_documents(operator_features)

        return [
            {
                "similarity_score": self.calculate_similarity(operator_features, doc),
                "operator_name": doc.metadata["operator_name"],
                "operator_type": doc.metadata["operator_type"],
                "file_path": doc.metadata["file_path"],
                "description": doc.page_content
            }
            for doc in docs
        ]
    
    def sample(self):
        """
        按照特定标准筛选算子优化方案
        """
        pass
    
    def insert(self, parsed_code: ParsedCode, op_name, arch, framework, impl_type):
        """
        插入新的算子调度方案
        """
        operator_path = Path(database_path) / "operators"

        # 创建框架实现文件
        framework_file = operator_path / arch / op_name / f"{framework}.py"
        with open(framework_file, "w", encoding="utf-8") as f:
            f.write(self.framework_code)

        # 创建具体实现文件
        if impl_type == "triton":
            impl_code = parsed_code.triton_code
        elif impl_type == "swft":
            impl_code = parsed_code.swft_code
        impl_file = operator_path / arch / op_name / f"{impl_type}.py"
        with open(impl_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

    def update(self):
        """
        更新已有算子调度方案
        """
        pass

    def delete(self, op_name, arch, type=""):
        """
        删除算子调度方案
        """
        operator_path = Path(database_path) / "operators"
        file_path = operator_path / arch / op_name
        if type:
            file_path = file_path / f"{type}.py"
            file_path.unlink()
        else:
            file_path.rmdir()


# 使用示例
if __name__ == "__main__":
    # 初始化系统
    db_system = DatabaseRAG("./rag_config.yaml")

    # 准备查询特征
    query_features = {
        "type": "reduce+elementwise融合",
        "name": "custom_softmax",
        "shape": "reduce轴:64, 非reduce轴:8192",
        "description": "包含exp和sum操作的融合算子",
        "arch": "ascend910b4"
    }

    # 检索优化方案
    results = db_system.retrieve_optimization_plans(query_features)

    # 输出结果
    print(f"找到 {len(results)} 个匹配的优化方案:")
    for i, res in enumerate(results, 1):
        print(f"\n#{i} 相似度: {res['similarity_score']:.4f}")
        print(f"算子名称: {res['operator_name']}")
        print(f"文件路径: {res['file_path']}")
        print(f"特征描述: {res['description'][:100]}...")
