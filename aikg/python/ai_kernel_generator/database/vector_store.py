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
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

class VectorStore:
    """
    基于RAG的优化方案检索器，检索最相似的算子调度方案
    """
    def __init__(self, database_path: str, embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", index_name: str = "vector_store"):
        os.environ["OMP_NUM_THREADS"] = "8"
        self.database_path = database_path
        self.index_path = str(Path(self.database_path) / index_name)
        self.enable_vector_store = True
        self.embedding_model = self._load_embedding_model(embedding_model_name) 
        self.vector_store = self._load_or_create_store()
        
    def _load_embedding_model(self, embedding_model_name: str = "GanymedeNil/text2vec-large-chinese"):
        """从配置文件加载嵌入模型"""
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        def load_huggingface_embedding_model(model_name: str):
            """加载HuggingFace嵌入模型"""
            try:
                embedding = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},  # 如有GPU可改为'cuda'
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
    
    def _load_or_create_store(self):
        """加载或创建向量存储"""
        if not self.enable_vector_store:
            return None

        index_path = Path(self.index_path)
        
        # 如果索引不存在则创建
        if not (index_path / "index.faiss").exists():
            logger.info("Building operator feature vector database...")
            return self._build_vector_store()
        
        # 加载现有索引
        logger.info("Loading existing vector index...")
        return FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # 注意安全性
        )
    
    def _build_vector_store(self):
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
            
            backend = metadata.get('backend', '')
            arch = metadata.get('arch', '')
            dsl = metadata.get('dsl', '')
            feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)
            # 创建检索文档
            doc = Document(
                page_content=", ".join([f"{k}: {v}" for k, v in metadata.items() if k not in {'backend', 'arch', 'dsl', 'profile'}]),
                metadata={
                    "file_path": str(op_subdir),
                    "feature_invariants": feature_invariants
                }
            )
            documents.append(doc)
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
            fetch_k = max(20, 5 * k),
            lambda_mult = 0.2, # 极致多样性
            filter={"feature_invariants": feature_invariants}, 
        )
        
    def similarity_search(self, query: str, feature_invariants: str, k: int = 5, fetch_k: int = 20):
        "执行语义搜索并返回匹配的文档"
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            fetch_k = max(fetch_k, 5 * k),
            filter={"feature_invariants": feature_invariants}
        )
        
    def similarity_search_with_score(self, query: str, feature_invariants: str, k: int = 5, fetch_k: int = 20):
        "执行语义搜索并返回匹配的文档及其相似度得分"
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            fetch_k = max(fetch_k, 5 * k),
            filter={"feature_invariants": feature_invariants}
        )

    def insert(self, backend: str, arch: str, dsl: str, md5_hash: str):
        """向向量存储添加新的算子特征文档"""
        if not self.enable_vector_store:
            return

        metadata_path = Path(self.database_path) / "operators" / arch / dsl / md5_hash / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError(f"算子元数据文件 {str(metadata_path)} 不存在")
        
        with open(str(metadata_path), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 获取算子目录
        op_dir = metadata_path.parent

        # 创建文档对象
        feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)
        # 创建检索文档
        doc = Document(
            page_content=", ".join([f"{k}: {v}" for k, v in metadata.items() if k not in {'backend', 'arch', 'dsl', 'profile'}]),
            metadata={
                "file_path": str(op_dir),
                "feature_invariants": feature_invariants
            }
        )

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