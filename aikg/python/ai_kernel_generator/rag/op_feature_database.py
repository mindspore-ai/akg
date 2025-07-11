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

import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class OpFeatureDatabase:
    """
    基于RAG的优化方案检索器，检索最相似的算子调度方案
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding_model = self._load_embedding_model()
        self.vector_store = self._load_or_create_vector_store()

    def _load_embedding_model(self):
        """加载嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=self.config['embedding_model'],
            model_kwargs={'device': 'cpu'},  # 如有GPU可改为'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )

    def _load_or_create_vector_store(self):
        """加载或创建向量存储"""
        index_path = Path(self.config['index_path'])

        # 如果索引不存在则创建
        if not (index_path / "index.faiss").exists():
            print("构建算子特征向量库...")
            return self._build_vector_store()

        # 加载现有索引
        print("加载现有向量索引...")
        return FAISS.load_local(
            folder_path=str(index_path),
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # 注意安全性
        )

    def _build_vector_store(self):
        """从算子元数据构建向量存储"""
        operator_path = Path(self.config['operator_path'])
        documents = []

        # 遍历算子目录收集元数据
        for op_dir in operator_path.iterdir():
            if not op_dir.is_dir():
                continue

            metadata_file = op_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            # 加载算子元数据
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # 创建检索文档
            doc = Document(
                page_content=metadata['description'],
                metadata={
                    "operator_name": op_dir.name,
                    "operator_type": metadata.get('type', ''),
                    "operator_shape": metadata.get('shape', ''),
                    "file_path": str(next(op_dir.glob("*.py"))),  # 查找py方案文件
                }
            )
            documents.append(doc)

        # 创建向量存储
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

        # 保存索引
        vector_store.save_local(self.config['index_path'])
        return vector_store
