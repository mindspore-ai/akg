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
from typing import List, Any, Dict
from langchain_core.documents import Document
from akg_agents.database.vector_store import VectorStore
from akg_agents.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)


class CoderVectorStore(VectorStore):
    """
    算子代码向量存储，继承自 VectorStore 基类。
    专门用于存储和检索算子实现代码相关的向量。
    
    Attributes:
        features: 用于构建文档的算子特征列表，默认包含 op_name, op_type, input_specs, output_specs, computation
    """
    # 算子代码相关的默认特征
    DEFAULT_FEATURES = ["op_name", "op_type", "input_specs", "output_specs", "computation"]
    
    # 单例模式实现
    _instances: Dict[str, 'CoderVectorStore'] = {}
    _lock = False
    
    def __new__(cls, 
                database_path: str, 
                embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", 
                index_name: str = "coder_vector_store",
                features: List[str] = None,
                config: dict = None):
        # 使用数据库路径和索引名称作为实例的唯一标识
        instance_key = get_md5_hash(database_path=database_path, index_name=index_name)
        
        if instance_key not in cls._instances or cls._instances[instance_key] is None:
            while cls._lock:
                pass
            cls._lock = True
            try:
                if instance_key not in cls._instances or cls._instances[instance_key] is None:
                    cls._instances[instance_key] = super(CoderVectorStore, cls).__new__(cls)
            finally:
                cls._lock = False
        
        return cls._instances[instance_key]
    
    def __init__(self, 
                 database_path: str, 
                 embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", 
                 index_name: str = "coder_vector_store",
                 features: List[str] = None,
                 config: dict = None):
        # 防止重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # 使用算子特定的默认特征
        self.features = features if features is not None else self.DEFAULT_FEATURES
        super().__init__(database_path, embedding_model_name, index_name, config)
        self._initialized = True
    
    @classmethod
    def clear_instances(cls):
        """清除单例实例缓存"""
        cls._instances.clear()
    
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
                "op_type": metadata.get('op_type', ''),
                "file_path": file_path,
                "feature_invariants": feature_invariants
            }
        )
        
        return doc
