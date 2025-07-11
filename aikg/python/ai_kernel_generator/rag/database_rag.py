import os
import yaml
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ai_kernel_generator.rag.op_feature_database import OpFeatureDatabase
from typing import Optional


class DatabaseRAG(BaseRetriever):
    """算子优化方案RAG数据库系统"""
    # 必须显式声明所有字段
    config: dict
    retriever: OpFeatureDatabase

    def __init__(self, config_path: str = "rag_config.yaml"):
        # 先加载配置
        config = self._load_config(config_path)

        # 初始化检索器 (在基类初始化之前)
        retriever = OpFeatureDatabase(config)

        # 初始化基类并提供所有声明字段的值
        super().__init__(
            config=config,
            retriever=retriever
        )

    @staticmethod
    def _load_config(config_path):
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 设置默认路径
        base_path = Path(__file__).parent.resolve()
        config['operator_path'] = str(base_path / config.get('operator_path', 'database/operators'))
        config['index_path'] = str(base_path / config.get('index_path', 'database/vector_store'))
        return config

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        """实现基类要求的抽象方法"""
        # 使用正确的相似度搜索接口
        return self.retriever.vector_store.similarity_search(
            query=query,
            k=self.config['top_k']
        )

    def retrieve_optimization_plans(self, operator_features: str):
        """
        根据算子特征检索优化方案（对外接口）
        operator_features: 包含算子特征的字符串
        """
        query = self._format_features_query(operator_features)

        # 使用基类提供的标准方法检索文档
        docs = self.get_relevant_documents(query)

        return [
            {
                "similarity_score": self._calculate_similarity(query, doc),
                "operator_name": doc.metadata["operator_name"],
                "operator_type": doc.metadata["operator_type"],
                "file_path": doc.metadata["file_path"],
                "description": doc.page_content
            }
            for doc in docs
        ]

    def _format_features_query(self, features: str):
        """将特征字典转换为自然语言查询"""
        if isinstance(features, dict):
            return (f"算子类型: {features.get('type')}, "
                    f"算子名称: {features.get('name')}, "
                    f"算子形态: {features.get('shape')}, "
                    f"描述: {features.get('description')}")
        return features

    def _calculate_similarity(self, query: str, document: Document):
        """计算余弦相似度"""
        # 获取查询和文档的嵌入向量
        query_embed = self.retriever.embedding_model.embed_query(query)
        doc_embed = self.retriever.embedding_model.embed_query(document.page_content)

        # 计算余弦相似度
        dot_product = sum(q * d for q, d in zip(query_embed, doc_embed))
        norm_q = sum(q * q for q in query_embed) ** 0.5
        norm_d = sum(d * d for d in doc_embed) ** 0.5

        # 避免除以零
        return dot_product / (norm_q * norm_d + 1e-10)


# 使用示例
if __name__ == "__main__":
    # 初始化系统
    db_system = DatabaseRAG("./rag_config.yaml")

    # 准备查询特征
    query_features = {
        "type": "reduce+elementwise融合",
        "name": "custom_softmax",
        "shape": "reduce轴:64, 非reduce轴:8192",
        "description": "包含exp和sum操作的融合算子"
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
