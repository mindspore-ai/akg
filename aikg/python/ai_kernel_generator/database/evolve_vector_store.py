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
from typing import List
from langchain_core.documents import Document
from ai_kernel_generator.database.vector_store import VectorStore

logger = logging.getLogger(__name__)

class EvolveVectorStore(VectorStore):
    """
    基于RAG的进化算子调度方案检索器，继承自VectorStore并专门处理schedule相关功能
    """
    
    def __init__(self, 
                 database_path: str, 
                 embedding_model_name: str = "GanymedeNil/text2vec-large-chinese", 
                 index_name: str = "evolve_vector_store",
                 features: List[str] = ["base", "pass", "text"]):
        super().__init__(database_path, embedding_model_name, index_name, features)
    
    @staticmethod
    def get_page_content(metadata: dict, features: List[str]):
        """
        从算子元数据生成特定特征的页面内容
        page_content为用于向量化存储的文本内容。
        
        Args:
            metadata: 算子元数据字典
            features: 需要提取的特征列表
            
        Returns:
            str: 格式化的特征文本内容
        """
        page_content_parts = []
        for schedule_block in features:
            # 处理schedule块字段，直接展开为键值对
            block = metadata.get('schedule', {}).get(schedule_block, {})
            if isinstance(block, dict):
                for key, value in block.items():
                    page_content_parts.append(f"{key}: {value}")
        return ", ".join(page_content_parts)
    
    def gen_document(self, metadata: dict, file_path: str):
        """从算子元数据生成文档，支持schedule块字段"""
        page_content = self.get_page_content(metadata, self.features)
        return Document(
            page_content=page_content,
            metadata={"file_path": file_path}
        )
    
    def max_marginal_relevance_search(self, query: str, k: int = 5):
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            lambda_mult=0.1 # 极致多样性
        )