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
from typing import List, Dict, Any
from langchain_core.documents import Document
from ai_kernel_generator.database.vector_store import VectorStore
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)


class CoderVectorStore(VectorStore):
    """
    基于RAG的代码生成器向量存储，继承自VectorStore
    专门用于代码生成相关的向量检索
    """
    # 单例模式实现 - 重写父类的单例管理
    _instances: Dict[str, 'CoderVectorStore'] = {}
    _lock = False  # 简单的锁机制避免并发问题
    
    def __new__(cls, 
                database_path: str, 
                embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", 
                index_name: str = "vector_store",
                features: List[str] = ["op_name", "op_type", "input_specs", "output_specs", "computation"]):
        # 使用数据库路径、索引名称和特征列表的组合作为实例的唯一标识
        instance_key = get_md5_hash(database_path=database_path, index_name=index_name, features=features)
        
        # 检查实例是否已存在
        if instance_key not in cls._instances or cls._instances[instance_key] is None:
            # 简单锁机制
            while cls._lock:
                pass
            cls._lock = True
            try:
                # 双重检查锁定模式
                if instance_key not in cls._instances or cls._instances[instance_key] is None:
                    cls._instances[instance_key] = super(CoderVectorStore, cls).__new__(cls)
            finally:
                cls._lock = False
        
        return cls._instances[instance_key]
    
    def __init__(self, 
                 database_path: str, 
                 embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", 
                 index_name: str = "coder_vector_store",
                 features: List[str] = ["op_name", "op_type", "input_specs", "output_specs", "computation"]):
        super().__init__(database_path, embedding_model_name, index_name, features)
    
    def gen_document(self, metadata: dict, file_path: str, other_args: Any = None):
        """从代码元数据生成文档 - 重写父类方法"""
        backend = metadata.get('backend', '')
        arch = metadata.get('arch', '')
        dsl = metadata.get('dsl', '')
        feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)
        
        # 构建页面内容，包含代码相关的特征
        page_content_parts = []
        for feature in self.features:
            if feature in metadata:
                page_content_parts.append(f"{feature}: {metadata[feature]}")
        
        doc = Document(
            page_content=", ".join(page_content_parts),
            metadata={
                "code_type": metadata.get('code_type', ''),
                "function_name": metadata.get('function_name', ''),
                "file_path": file_path,
                "feature_invariants": feature_invariants
            }
        )
        
        return doc
