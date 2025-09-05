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

import logging
from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from ai_kernel_generator.database.vector_store import VectorStore
from ai_kernel_generator.database.database import Database, RetrievalStrategy
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

DEFAULT_CODER_DATABASE_PATH = Path(get_project_root()).parent.parent / "coder_database"

class CoderDatabase(Database):
    # 单例模式实现
    _instances: Dict[str, 'CoderDatabase'] = {}
    _lock = False  # 简单的锁机制避免并发问题
    
    def __new__(cls, database_path: str = "", config: dict = None):
        database_path = database_path or str(DEFAULT_CODER_DATABASE_PATH)
        # 使用数据库路径作为实例的唯一标识
        instance_key = get_md5_hash(database_path=database_path)
        
        # 检查实例是否已存在
        if instance_key not in cls._instances or cls._instances[instance_key] is None:
            # 简单锁机制
            while cls._lock:
                pass
            cls._lock = True
            try:
                # 双重检查锁定模式
                if instance_key not in cls._instances or cls._instances[instance_key] is None:
                    cls._instances[instance_key] = super(CoderDatabase, cls).__new__(cls)
            finally:
                cls._lock = False
        
        return cls._instances[instance_key]
        
    def __init__(self, database_path: str = "", config: dict = None):
        # 防止重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.database_path = database_path or str(DEFAULT_CODER_DATABASE_PATH)
        self.computation_vector_store = VectorStore(
            database_path=self.database_path,
            index_name="computation_vector_store",
            features=["op_name", "computation"]
        )
        self.vector_stores = [self.computation_vector_store]
        super().__init__(self.database_path, self.vector_stores, config)
        self._initialized = True
    
    def hierarchy_search(self, features: dict, feature_invariants: str, k: int = 5):
        """层次检索：先按计算逻辑检索，再按shape检索"""
        op_type = features["op_type"]
        computation_query = ", ".join([f"{key}: {features[key]}" for key in self.computation_vector_store.features])
        computation_docs = self.computation_vector_store.vector_store.similarity_search(
            query=computation_query, 
            k=max(20, 5 * k),
            fetch_k=max(100, 20 * k),
            filter={"feature_invariants": feature_invariants, "op_type": op_type}
        )
        shape_features = ["input_specs", "output_specs"]
        shape_docs = []
        for doc in computation_docs:
            doc_path = doc.metadata.get("file_path", "")
            doc_content = self.get_case_content(shape_features, doc_path)
            shape_doc = Document(
                page_content=", ".join([f"{key}: {doc_content[key]}" for key in shape_features]),
                metadata={"file_path": doc_path}
            )
            shape_docs.append(shape_doc)
        shape_vector_store = FAISS.from_documents(shape_docs, self.computation_vector_store.embedding_model)
        docs = shape_vector_store.similarity_search(
            query=", ".join([f"{key}: {features[key]}" for key in shape_features]),
            k=k
        )
        return docs


    async def samples(self, output_content:List[str], sample_num:int = 5, impl_code: str = "", framework_code:str = "",
                      backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        """
        Evolve采样方案，根据当前算子的特征信息，从数据库中采样出优化性和随机性的算子实现。
        """
        need_extract_features = False
        for vector_store in self.vector_stores:
            if vector_store.enable_vector_store:
                need_extract_features = True
                break
        
        if need_extract_features:
            features = await self.extract_features(impl_code, framework_code, backend, arch, dsl)
            feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)
            
            docs = self.hierarchy_search(features, feature_invariants, sample_num)
            result = self.get_output_content(output_content, RetrievalStrategy.HIERARCHY, docs, dsl, framework)
        else:
            random_res = self.randomicity_search(output_content, sample_num, backend, arch, dsl, framework)
            result = random_res
        
        return result
