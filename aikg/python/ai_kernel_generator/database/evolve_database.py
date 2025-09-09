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
import logging
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from ai_kernel_generator.database.evolve_vector_store import EvolveVectorStore
from ai_kernel_generator.database.database import Database, RetrievalStrategy
from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)

DEFAULT_EVOLVE_DATABASE_PATH = Path(get_project_root()).parent.parent / "evolve_database"

class EvolveDatabase(Database):
    def __init__(self, database_path: str = "", config: dict = None):
        self.database_path = database_path or str(DEFAULT_EVOLVE_DATABASE_PATH)
        self.base_vector_store = EvolveVectorStore(
            database_path=self.database_path,
            index_name="base_vector_store",
            features=["schedule.base"]
        )
        self.pass_vector_store = EvolveVectorStore(
            database_path=self.database_path,
            index_name="pass_vector_store",
            features=["schedule.PASS"]
        )
        self.text_vector_store = EvolveVectorStore(
            database_path=self.database_path,
            index_name="text_vector_store",
            features=["schedule.text"]
        )
        self.vector_stores = [self.base_vector_store, self.pass_vector_store, self.text_vector_store]
        self.vector_stores_map = {
            self.base_vector_store: "base",
            self.pass_vector_store: "pass",
            self.text_vector_store: "text"
        }
        super().__init__(self.database_path, self.vector_stores, config)
    
    def max_marginal_relevance_rerank(self, query: str, docs: List[Document], k: int):
        """
        使用MMR算法对文档进行重排
        """
        vector_store = FAISS.from_documents(docs, self.base_vector_store.embedding_model)
        return vector_store.max_marginal_relevance_search(
            query=query, 
            k=k,
            lambda_mult=0.2
        )

    def optimality_search(self):
        """遍历数据库路径查找profile最小的metadata.json文件"""
        min_profile = float('inf')
        min_path = None
        
        for metadata_file in Path(self.database_path).rglob('metadata.json'):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    profile = data.get('profile', float('inf'))
                    if profile < min_profile:
                        min_profile = profile
                        min_path = str(metadata_file.parent)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"解析{metadata_file}失败: {e}")
        
        return [Document(page_content="", metadata={"file_path": min_path})] if min_path else []
    
    def fusion_search(self, features: dict, k: int, similar: List[str] = ["base"], diverse: List[str] = ["pass", "text"], rrf_k: int = 60):
        """
        多向量混合检索方法，RRF融合，MMR重排
        
        Args:
            features: str, 特征字典
            k: int, 最终返回的文档数量
            similar: List[str], 希望相似的调度
            diverse: List[str], 希望多样性的调度
            rrf_k: int, RRF算法中的k参数，用于计算倒数排名分数
            
        Returns:
            List[Document]: 经过RRF融合、MMR重排后的文档列表
        """
        if not features:
            logger.warning("Features is empty, returning empty results")
            return []
        
        # 检查参数有效性
        valid_types = set(self.vector_stores_map.values())  # {"base", "pass", "text"}
        similar_lower = [s.lower() for s in similar]
        diverse_lower = [d.lower() for d in diverse]
        for param_name, param_values in [("similar", similar_lower), ("diverse", diverse_lower)]:
            invalid = [v for v in param_values if v not in valid_types]
            if invalid:
                raise ValueError(f"Invalid values in '{param_name}' parameter: {invalid}. "
                               f"Valid values are: {list(valid_types)}")
        if overlap := set(similar_lower) & set(diverse_lower):
            raise ValueError(f"Found overlapping values between 'similar' and 'diverse': {overlap}")

        # 存储每个向量库的检索结果
        docs_list = []
        for vector_store in self.vector_stores:
            try:
                if self.vector_stores_map[vector_store] in similar_lower:
                    docs = vector_store.vector_store.similarity_search(
                        query=vector_store.get_page_content(features, vector_store.features),
                        k=max(k * 3, 30)  # 获取更多候选文档用于融合
                    )
                    docs_list.append(docs)
            except Exception as e:
                logger.error(f"Error searching vector store {self.vector_stores_map[vector_store]}: {e}")
                raise e
        
        if not docs_list:
            logger.warning("No vector stores returned results")
            return []
        
        fused_docs = self._reciprocal_rank_fusion(docs_list, max(k * 2, 10), rrf_k)

        query = EvolveVectorStore.get_page_content(features, diverse_lower)
        mmr_docs = self.max_marginal_relevance_rerank(query, fused_docs, k)
        return mmr_docs
    
    def _reciprocal_rank_fusion(self, docs_list: List[List], k: int, rrf_k: int = 60):
        """
        使用RRF算法融合多个检索结果
        
        Args:
            docs_list: 多个向量库的检索结果列表
            k: 最终返回的文档数量
            rrf_k: RRF算法中的k参数
            
        Returns:
            List[Document]: 融合重排后的文档列表
        """
        # 创建文档到分数的映射
        doc_scores = {}
        
        # 为每个向量库的结果计算RRF分数
        for docs in docs_list:
            if not docs:
                continue
                
            for rank, doc in enumerate(docs):
                # 使用文档的file_path作为唯一标识符
                doc_id = doc.metadata.get("file_path", "")
                if not doc_id:
                    continue
                
                # 计算RRF分数: 1 / (rrf_k + rank)
                rrf_score = 1.0 / (rrf_k + rank + 1)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]["score"] += rrf_score
                else:
                    file_path = doc.metadata.get("file_path", "")
                    metadata_file = Path(file_path) / "metadata.json"
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    full_doc = Document(
                        page_content=EvolveVectorStore.get_page_content(metadata, ["schedule"]), 
                        metadata={"file_path": file_path}
                    )
                    doc_scores[doc_id] = {
                        "doc": full_doc,
                        "score": rrf_score,
                        "ranks": [rank + 1]  # 记录在每个向量库中的排名
                    }
        
        if not doc_scores:
            return []
        
        # 按RRF分数降序排序
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # 返回前sample_num个文档
        result_docs = []
        sample_num = min(k, len(sorted_docs))
        for item in sorted_docs[:sample_num]:
            # 添加融合信息到metadata
            doc = item["doc"]
            doc.metadata["fusion_score"] = item["score"]
            doc.metadata["fusion_ranks"] = item["ranks"]
            result_docs.append(doc)
        return result_docs

    async def samples(self, output_content:List[str], sample_num:int = 5, impl_code: str = "", framework_code:str = "",
                      backend: str = "", arch: str = "", dsl: str = "", framework: str = "", similar: List[str] = ["base"]):
        """
        Evolve采样方案，根据当前算子的特征信息，从数据库中采样出优化性和随机性的算子实现。
        
        Args:
            output_content: 输出内容列表
            sample_num: 采样数量
            impl_code: 实现代码
            framework_code: 框架代码
            backend: 后端
            arch: 架构
            dsl: DSL
            framework: 框架
            similar: 希望相似的调度
        """
        result = []
        optimality_docs = self.optimality_search()
        optimality_res = self.get_output_content(output_content, RetrievalStrategy.OPTIMALITY, optimality_docs, dsl, framework)
        result.extend(optimality_res)
        
        need_extract_features = False
        for vector_store in self.vector_stores:
            if vector_store.enable_vector_store:
                need_extract_features = True
                break
        
        if need_extract_features:
            features = await self.extract_features(impl_code, framework_code, backend, arch, dsl)
            # TODO:使用多向量融合检索
            # docs = self.fusion_search(features, sample_num - 1, similar)
            # schedule_res = self.get_output_content(output_content, RetrievalStrategy.FUSION, docs, dsl, framework)

            # 切分调度多样性检索
            query = EvolveVectorStore.get_page_content(features, self.base_vector_store.features)
            docs = self.base_vector_store.max_marginal_relevance_search(query, sample_num)
            schedule_res = self.get_output_content(output_content, RetrievalStrategy.MMR, docs, dsl, framework)
            
            result.extend(schedule_res)
        else:
            random_res = self.randomicity_search(output_content, sample_num - 1, backend, arch, dsl, framework)
            result.extend(random_res)
        
        return result
