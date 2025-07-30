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
from pathlib import Path
import json
from langchain_core.documents import Document
from ai_kernel_generator.database.database import Database, RetrievalStrategy
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

DEFAULT_EVOLVE_DATABASE_PATH = Path(get_project_root()).parent.parent / "evolve_database"

class EvolveDatabase(Database):
    def __init__(self, config_path: str = "", database_path: str = ""):
        super().__init__(config_path, database_path)
        self.database_path = database_path or str(DEFAULT_EVOLVE_DATABASE_PATH)

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

    async def samples(self, output_content:List[str], sample_num:int = 5, impl_code: str = "", framework_code:str = "",
                      backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        """
        Evolve采样方案，根据当前算子的特征信息，从数据库中采样出优化性和随机性的算子实现。
        """
        features = await self.feature_extractor(impl_code, framework_code, backend, arch, dsl)
        features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
        feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)

        result = []
        optimality_docs = self.optimality_search()
        optimality_res = self.get_output_content(output_content, RetrievalStrategy.OPTIMALITY, optimality_docs, dsl, framework)
        
        random_docs = self.randomicity_search(features_str, feature_invariants, sample_num - 1)
        random_res = self.get_output_content(output_content, RetrievalStrategy.RANDOMICITY, random_docs, dsl, framework)
        
        result.extend(optimality_res)
        result.extend(random_res)
        return result
