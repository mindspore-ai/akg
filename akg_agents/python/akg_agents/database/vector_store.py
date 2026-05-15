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
from akg_agents.utils.common_utils import get_md5_hash
from akg_agents.core_v2.llm import create_embedding_model

logger = logging.getLogger(__name__)


def _restore_root_logger_level():
    """
    恢复根 logger 的日志级别。
    
    FAISS/langchain 在初始化时会将根 logger 的 level 从 INFO 改为 WARNING，
    导致其他模块的 INFO 日志无法输出。此函数在 FAISS 操作后恢复日志级别。
    """
    from akg_agents import log_level
    
    root_logger = logging.getLogger()
    if root_logger.level > log_level:
        root_logger.setLevel(log_level)

def _auto_detect_device() -> str:
    """自动检测可用的设备"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

class VectorStore(ABC):
    """
    向量存储抽象基类，定义了向量存储的基本框架。
    子类需要实现 gen_document 方法来定义如何从元数据生成文档。
    
    注意：基类不实现单例模式，由子类根据需要实现自己的单例模式。
    """
        
    def __init__(self, 
                 database_path: str, 
                 embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", 
                 index_name: str = "vector_store",
                 config: dict = None):
        """
        初始化向量存储
        
        Args:
            database_path: 数据库路径
            embedding_model_name: 嵌入模型名称
            index_name: 索引名称
            config: 配置字典
        """
        os.environ["OMP_NUM_THREADS"] = "8"
        self.database_path = database_path
        self.index_name = index_name
        self.index_path = str(Path(self.database_path) / index_name)
        self.config = config or {}
        self.enable_vector_store = True
        self.embedding_model = self._load_embedding_model(embedding_model_name) 
        self.is_exist = (Path(self.index_path) / "index.faiss").exists()
        self.vector_store = self._load_or_create_store(None)

    def _load_embedding_model(self, embedding_model_name: str = "GanymedeNil/text2vec-large-chinese"):
        """加载嵌入模型
        
        优先级：
        1. 远程 API（通过 create_embedding_model，自动检查环境变量）
        2. 本地 HuggingFace 模型
        """
        # 【最高优先级】尝试使用远程 embedding API（自动检查环境变量）
        try:
            embedding = create_embedding_model()
            logger.info("Using remote embedding API")
            return embedding
        except Exception as e:
            logger.info(f"Remote embedding API not available: {e}, trying local model")
        
        # 【次优先级】使用本地 HuggingFace 模型
        logger.info(f"Loading local embedding model: {embedding_model_name}")
        
        def load_huggingface_embedding_model(model_name: str):
            """加载HuggingFace嵌入模型"""
            try:
                embedding = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': _auto_detect_device()},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # HuggingFaceEmbeddings 初始化可能会影响日志配置，恢复之
                _restore_root_logger_level()
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace model: {model_name}, error: {e}")
                return None
        
        # 优先使用本地模型路径
        local_model_path = os.path.join(os.path.expanduser("~"), ".akg_agents", "text2vec-large-chinese")
        if os.path.exists(local_model_path):
            logger.info(f"Trying to load local embedding model from {local_model_path}")
            embedding = load_huggingface_embedding_model(local_model_path)
            if embedding:
                return embedding
            logger.warning(f"Failed to load local embedding model from {local_model_path}")
        else:
            logger.info(f"Local embedding model not found at {local_model_path}")
        
        # 所有加载尝试都失败，抛出异常
        error_msg = (
            f"Failed to load embedding model '{embedding_model_name}'. "
            f"Please either:\n"
            f"  1. Set environment variables: AKG_AGENTS_EMBEDDING_BASE_URL, AKG_AGENTS_EMBEDDING_MODEL_NAME, AKG_AGENTS_EMBEDDING_API_KEY\n"
            f"  2. Download the local model by running: bash download.sh --with_local_model"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _load_or_create_store(self, other_args: Any = None):
        """加载或创建向量存储"""
        if not self.enable_vector_store:
            return None

        index_path = Path(self.index_path)
        
        # 如果索引不存在则创建
        if not self.is_exist:
            logger.info(f"Building vector database: {index_path.name}...")
            return self._build_vector_store(other_args)
        
        # 加载现有索引
        logger.info(f"Loading existing vector index: {index_path.name}...")
        result = FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # 注意安全性
        )
        # FAISS.load_local 可能会影响日志配置，恢复之
        _restore_root_logger_level()
        return result

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
        """从元数据构建向量存储"""
        root_dir = Path(self.database_path)
        documents = []
        
        # 递归查找所有metadata.json文件（支持任意目录结构）
        for metadata_file in root_dir.rglob("metadata.json"):
            op_subdir = metadata_file.parent  # 元数据文件所在目录
            
            # 跳过隐藏目录中的文件
            if any(part.startswith('.') for part in op_subdir.parts):
                continue
                
            # 加载元数据
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 创建检索文档
            documents.append(self.gen_document(metadata, str(op_subdir), other_args))
        # 创建向量存储
        if not documents:
            # 创建空向量存储
            dummy_doc = Document(page_content="", metadata={})
            vector_store = FAISS.from_documents([dummy_doc], self.embedding_model)
            # FAISS.from_documents 可能会影响日志配置，恢复之
            _restore_root_logger_level()
            # 删除 dummy 文档
            dummy_id = list(vector_store.index_to_docstore_id.values())[0]
            vector_store.delete([dummy_id])
        else:
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model,
            )
            # FAISS.from_documents 可能会影响日志配置，恢复之
            _restore_root_logger_level()
        
        # 保存索引
        vector_store.save_local(self.index_path)
        return vector_store
    
    def max_marginal_relevance_search(self, query: str, k: int = 5):
        "执行最大边际相关搜索并返回匹配的文档"
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=max(20, 5 * k),
            lambda_mult=0.2, # 极致多样性
        )
        
    def similarity_search(self, query: str, k: int = 5, fetch_k: int = 20):
        "执行语义搜索并返回匹配的文档"
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            fetch_k=max(fetch_k, 5 * k)
        )
        
    def similarity_search_with_score(self, query: str, k: int = 5, fetch_k: int = 20):
        "执行语义搜索并返回匹配的文档及其相似度得分"
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            fetch_k=max(fetch_k, 5 * k)
        )
        
    def insert(self, doc_path: str):
        """向向量存储添加新的文档"""
        if not self.enable_vector_store:
            return

        metadata_path = Path(self.database_path) / doc_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"元数据文件 {str(metadata_path)} 不存在")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        op_dir = metadata_path.parent
        doc = self.gen_document(metadata, str(op_dir))
        self.delete(doc_path)
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(self.index_path)
        logger.info(f"Successfully added document with path={doc_path} to vector index")

    def delete(self, doc_path: str):
        """从向量存储中删除指定文档"""
        if not self.enable_vector_store:
            return
        
        op_dir = str(Path(self.database_path) / doc_path)
        existing_ids = list(self.vector_store.index_to_docstore_id.values())
        for doc_id in existing_ids:
            existing_doc = self.vector_store.docstore.search(doc_id)
            if existing_doc.metadata.get("file_path") == op_dir:
                self.vector_store.delete([doc_id])
                self.vector_store.save_local(self.index_path)
                logger.info(f"Successfully removed document with path={doc_path} from vector index")
                return
        logger.info(f"Document with path={doc_path} not found in vector index")
    
    def has_doc(self, doc_path: str):
        """检查文档是否已存在"""
        if not self.enable_vector_store:
            return False
        op_dir = str(Path(self.database_path) / doc_path)
        existing_ids = list(self.vector_store.index_to_docstore_id.values())
        for doc_id in existing_ids:
            existing_doc = self.vector_store.docstore.search(doc_id)
            if existing_doc.metadata.get("file_path") == op_dir:
                return True
        return False
    
    def clear(self):
        """清空向量存储"""
        if not self.enable_vector_store:
            return

        self.vector_store.delete(list(self.vector_store.index_to_docstore_id.values()))
        self.vector_store.save_local(self.index_path)
        logger.info("Successfully cleared vector index")
    
    @classmethod
    def clear_instances(cls):
        """清除所有子类的单例实例缓存"""
        if hasattr(cls, '_instances'):
            cls._instances.clear()