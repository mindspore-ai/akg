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

import os
import json
import logging
from typing import Any, List, Dict
from pathlib import Path
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """
    基于RAG的优化方案检索器，检索最相似的算子调度方案
    抽象基类，定义了向量存储的基本框架，子类需要实现gen_document方法
    """
    # 单例模式实现
    _instances: Dict[str, 'VectorStore'] = {}
    _lock = False  # 简单的锁机制避免并发问题
    
    def __new__(cls, 
                database_path: str, 
                embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", 
                index_name: str = "vector_store",
                features: List[str] = None,
                config: dict = None):
        # 使用数据库路径、索引名称和特征列表的组合作为实例的唯一标识
        instance_key = get_md5_hash(database_path=database_path, index_name=index_name, features=features, config=config)
        
        # 检查实例是否已存在
        if instance_key not in cls._instances or cls._instances[instance_key] is None:
            # 简单锁机制
            while cls._lock:
                pass
            cls._lock = True
            try:
                # 双重检查锁定模式
                if instance_key not in cls._instances or cls._instances[instance_key] is None:
                    cls._instances[instance_key] = super(VectorStore, cls).__new__(cls)
            finally:
                cls._lock = False
        
        return cls._instances[instance_key]
        
    def __init__(self, 
                 database_path: str, 
                 embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", 
                 index_name: str = "vector_store",
                 features: List[str] = None,
                 config: dict = None):
        # 防止重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        os.environ["OMP_NUM_THREADS"] = "8"
        self.database_path = database_path
        self.features = features or ["op_name", "op_type", "input_specs", "output_specs", "computation", "schedule"]
        self.index_path = str(Path(self.database_path) / index_name)
        self.config = config or {}
        self.enable_vector_store = True
        self.embedding_model = self._load_embedding_model(embedding_model_name) 
        self.vector_store = self._load_or_create_store(None)
        self._initialized = True

    def _load_embedding_model(self, embedding_model_name: str = "GanymedeNil/text2vec-large-chinese"):
        """从配置文件加载嵌入模型"""
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        def load_huggingface_embedding_model(model_name: str):
            """加载HuggingFace嵌入模型"""
            try:
                embedding = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': self.config.get("database_config", {}).get("embedding_device", "cpu")},
                    encode_kwargs={'normalize_embeddings': True}
                )
                return embedding
            except Exception:
                logger.warning(f"Failed to load HuggingFace model: {model_name}")
                return None
        
        embedding = load_huggingface_embedding_model(embedding_model_name)
        if embedding:
            return embedding
        
        logger.warning("Automatic download of embedding model failed, using local embedding model")
        value = os.getenv("EMBEDDING_MODEL_PATH")  # 获取环境变量
        if value is None:
            logger.warning("EMBEDDING_MODEL_PATH environment variable not set")
        else:
            embedding = load_huggingface_embedding_model(value)
            if embedding:
                return embedding
            
        logger.warning("The embedding model was not found and the vector index library could not be enabled.")
        self.enable_vector_store = False
        return None
    
    def _load_or_create_store(self, other_args: Any = None):
        """加载或创建向量存储"""
        if not self.enable_vector_store:
            return None

        index_path = Path(self.index_path)
        
        # 如果索引不存在则创建
        if not (index_path / "index.faiss").exists():
            logger.info(f"Building operator feature vector database: {index_path.name}...")
            return self._build_vector_store(other_args)
        
        # 加载现有索引
        logger.info(f"Loading existing vector index: {index_path.name}...")
        return FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # 注意安全性
        )

    @abstractmethod
    def gen_document(self, metadata: dict, file_path: str, other_args: Any = None) -> Document:
        """
        从元数据生成文档 - 子类必须实现此方法
        
        Args:
            metadata: 元数据字典
            file_path: 文件路径
            other_args: 其他参数
            
        Returns:
            Document: 生成的文档对象
        """
        pass
    
    def _build_vector_store(self, other_args: Any = None):
        """从算子元数据构建向量存储"""
        root_dir = Path(self.database_path)
        documents = []
        
        # 递归查找所有metadata.json文件（支持任意目录结构）
        for metadata_file in root_dir.rglob("metadata.json"):
            op_subdir = metadata_file.parent  # 算子目录为元数据文件所在目录
            
            # 跳过隐藏目录中的文件
            if any(part.startswith('.') for part in op_subdir.parts):
                continue
                
            # 加载算子元数据
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 创建检索文档
            documents.append(self.gen_document(metadata, str(op_subdir), other_args))
        # 创建向量存储
        if not documents:
            # 创建空向量存储
            dummy_doc = Document(page_content="", metadata={})
            vector_store = FAISS.from_documents([dummy_doc], self.embedding_model)
            # 删除 dummy 文档
            dummy_id = list(vector_store.index_to_docstore_id.values())[0]
            vector_store.delete([dummy_id])
        else:
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model,
            )
        
        # 保存索引
        vector_store.save_local(self.index_path)
        return vector_store
    
    def max_marginal_relevance_search(self, query: str, feature_invariants: str, k: int = 5):
        "执行最大边际相关搜索并返回匹配的文档"
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=max(20, 5 * k),
            lambda_mult=0.2, # 极致多样性
            filter={"feature_invariants": feature_invariants}, 
        )
        
    def similarity_search(self, query: str, feature_invariants: str, k: int = 5, fetch_k: int = 20):
        "执行语义搜索并返回匹配的文档"
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            fetch_k=max(fetch_k, 5 * k),
            filter={"feature_invariants": feature_invariants}
        )
        
    def similarity_search_with_score(self, query: str, feature_invariants: str, k: int = 5, fetch_k: int = 20):
        "执行语义搜索并返回匹配的文档及其相似度得分"
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            fetch_k=max(fetch_k, 5 * k),
            filter={"feature_invariants": feature_invariants}
        )

    def insert(self, common_path: str, md5_hash: str):
        """向向量存储添加新的算子特征文档"""
        if not self.enable_vector_store:
            return

        metadata_path = Path(self.database_path) / common_path / md5_hash / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError(f"算子元数据文件 {str(metadata_path)} 不存在")
        
        with open(str(metadata_path), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 获取算子目录
        op_dir = metadata_path.parent

        # 创建文档对象
        doc = self.gen_document(metadata, str(op_dir))

        # 检查是否已存在相同算子的文档（去重并覆盖）
        self.delete(md5_hash)
        
        # 添加到向量存储并保存
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(self.index_path)
        logger.info(f"Successfully added operator with md5_hash={md5_hash} to vector index")

    def delete(self, md5_hash: str):
        """从向量存储中删除指定算子特征文档"""
        if not self.enable_vector_store:
            return

        existing_ids = list(self.vector_store.index_to_docstore_id.values())
        for doc_id in existing_ids:
            existing_doc = self.vector_store.docstore.search(doc_id)
            metadata_md5_hash = existing_doc.metadata.get("file_path").split('/')[-1]
            if metadata_md5_hash == md5_hash:
                # 已存在相同算子的文档，删除旧文档
                self.vector_store.delete([doc_id])
                self.vector_store.save_local(self.index_path)
                logger.info(f"Successfully removed operator with md5_hash={md5_hash} from vector index")
                return 
        logger.info(f"算子md5_hash={md5_hash}不存在于向量索引中")
    
    def clear(self):
        """清空向量存储"""
        if not self.enable_vector_store:
            return

        self.vector_store.delete(list(self.vector_store.index_to_docstore_id.values()))
        self.vector_store.save_local(self.index_path)
        logger.info("Successfully cleared vector index")