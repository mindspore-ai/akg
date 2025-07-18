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
import yaml
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.common_utils import get_md5_hash

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"
DEFAULT_INDEX_PATH = DEFAULT_DATABASE_PATH / "vector_store"

class VectorStore:
    """
    基于RAG的优化方案检索器，检索最相似的算子调度方案
    """
    def __init__(
        self, 
        config_path: str,
        database_path: str = "",
        index_path: str = ""):
        self.database_path = database_path if database_path else str(DEFAULT_DATABASE_PATH)
        self.index_path = index_path if index_path else str(DEFAULT_INDEX_PATH)
        self.config_path = config_path
        self.embedding_model = self.load_embedding_model()
        self.vector_store = self.load_or_create_vector_store()
        
    def load_embedding_model(self):
        """从配置文件加载嵌入模型"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        embedding_model = config.get('embedding_model', 'GanymedeNil/text2vec-large-chinese')
        return HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # 如有GPU可改为'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_or_create_vector_store(self):
        """加载或创建向量存储"""
        index_path = Path(self.index_path)
        
        # 如果索引不存在则创建
        if not (index_path / "index.faiss").exists():
            print("构建算子特征向量库...")
            return self.build_vector_store()
        
        # 加载现有索引
        print("加载现有向量索引...")
        return FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # 注意安全性
        )
    
    def build_vector_store(self):
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
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            arch = metadata.get('arch', '')
            impl_type = metadata.get('impl_type', '')
            backend = metadata.get('backend', '')
            feature_invariants = get_md5_hash(impl_type=impl_type, backend=backend, arch=arch)
            # 创建检索文档
            doc = Document(
                page_content=metadata['description'],
                metadata={
                    "operator_name": metadata.get('op_name', ''),
                    "operator_type": metadata.get('op_type', ''),
                    "operator_shape": metadata.get('op_axes_size', ''),
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
                embedding=self.embedding_model
            )
        
        # 保存索引
        vector_store.save_local(self.index_path)
        return vector_store

    def insert(self, arch: str, impl_type: str, md5_hash: str):
        """向向量存储添加新的算子特征文档
        Args:
            arch (str): 架构名称。
            impl_type (str): 实现类型。
            md5_hash (str): 哈希值
        """
        metadata_path = Path(self.database_path) / "operators" / arch / impl_type / md5_hash / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError(f"算子元数据文件 {str(metadata_path)} 不存在")
        
        with open(str(metadata_path), 'r') as f:
            metadata = json.load(f)
        
        # 获取算子目录
        op_dir = metadata_path.parent

        # 创建文档对象
        feature_invariants = get_md5_hash(impl_type=impl_type, backend=metadata.get('backend', ''), arch=arch)
        # 创建检索文档
        doc = Document(
            page_content=metadata['description'],
            metadata={
                "operator_name": metadata.get('op_name', ''),
                "operator_type": metadata.get('op_type', ''),
                "operator_shape": metadata.get('op_axes_size', ''),
                "file_path": str(op_dir),
                "feature_invariants": feature_invariants
            }
        )

        # 检查是否已存在相同算子的文档（去重并覆盖）
        self.delete(md5_hash)
        
        # 添加到向量存储并保存
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(self.index_path)
        print(f"成功添加算子md5_hash={md5_hash}到向量索引")

    def delete(self, md5_hash: str):
        existing_ids = list(self.vector_store.index_to_docstore_id.values())
        for doc_id in existing_ids:
            existing_doc = self.vector_store.docstore.search(doc_id)
            metadata_md5_hash = existing_doc.metadata.get("file_path").split('/')[-1]
            if metadata_md5_hash == md5_hash:
                # 已存在相同算子的文档，删除旧文档
                self.vector_store.delete([doc_id])
                self.vector_store.save_local(self.index_path)
                print(f"成功从向量索引中删除算子md5_hash={md5_hash}")
                return 
        print(f"算子md5_hash={md5_hash}不存在于向量索引中")