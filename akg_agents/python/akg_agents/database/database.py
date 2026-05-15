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

import shutil
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Any
from pathlib import Path
from akg_agents.database.vector_store import VectorStore
from akg_agents import get_project_root

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    RANDOMICITY = "randomicity"
    NAIVETY = "naivety"
    MMR = "max_marginal_relevance"
    OPTIMALITY = "optimality"
    RULE = "rule"
    HIERARCHY = "hierarchy"
    FUSION = "fusion"

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"

class Database(ABC):
    """
    数据库抽象基类，定义了数据库的基本框架。
    子类需要实现 _do_insert 和 _do_delete 方法来定义具体的增删逻辑。
    """
    def __init__(self, database_path: str = "", vector_stores: List[VectorStore] = [], config: dict = None):
        """初始化数据库系统
        
        Args:
            database_path: 数据库路径
            vector_stores: 向量存储列表
            config: 配置字典
        """
        self.database_path = database_path or str(DEFAULT_DATABASE_PATH)
        self.vector_stores = vector_stores
        
        if config:
            self.model_config = config.get("agent_model_config", {})
            self.config = config  # 保存完整 config
        else:
            raise ValueError("config is required for Database")

    async def _insert_with_vectors(self, doc_id: str, content: Any, mode: str = "skip", **kwargs):
        """插入文档的内部通用方法（供子类调用）
        
        Args:
            doc_id: 文档唯一标识
            content: 文档内容
            mode: 插入模式
                - 'skip': 如果文档已存在则跳过
                - 'overwrite': 如果文档已存在则覆盖
            **kwargs: 其他参数，传递给 _do_insert
            
        Returns:
            插入结果，由子类定义
        """
        if mode not in ('skip', 'overwrite'):
            raise ValueError("mode must be either 'skip' or 'overwrite'")
        
        file_path = Path(self.database_path) / doc_id
        
        if mode == 'skip' and file_path.exists():
            # 检查向量存储是否需要更新
            vector_store_to_insert = []
            for vector_store in self.vector_stores:
                if not vector_store.has_doc(doc_id):
                    vector_store_to_insert.append(vector_store)
            if not vector_store_to_insert:
                logger.debug(f"Document {doc_id} already exists, skipping")
                return None
        else:
            vector_store_to_insert = self.vector_stores
        
        # 调用子类实现的具体插入逻辑
        result = await self._do_insert(doc_id, content, file_path, **kwargs)
        
        # 更新向量存储
        for vector_store in vector_store_to_insert:
            vector_store.insert(doc_id)
        
        logger.info(f"Document inserted successfully, path: {file_path}")
        return result

    @abstractmethod
    async def _do_insert(self, doc_id: str, content: Any, file_path: Path, **kwargs):
        """执行具体的插入逻辑 - 子类必须实现此方法
        
        Args:
            doc_id: 文档唯一标识
            content: 文档内容
            file_path: 文件存储路径
            **kwargs: 其他参数
            
        Returns:
            插入结果
        """
        pass

    def _delete_with_vectors(self, doc_id: str):
        """删除文档的内部通用方法（供子类调用）
        
        Args:
            doc_id: 文档唯一标识
        """
        file_path = Path(self.database_path) / doc_id
        if not file_path.exists():
            logger.warning(f"Document does not exist: {file_path}")
            return
        
        # 调用子类实现的具体删除逻辑
        self._do_delete(doc_id, file_path)
        
        # 删除空的上级目录
        current_dir = file_path.parent
        while current_dir.exists() and current_dir != Path(self.database_path):
            if not any(current_dir.iterdir()):
                current_dir.rmdir()
                current_dir = current_dir.parent
            else:
                break
        
        # 从向量存储中删除
        for vector_store in self.vector_stores:
            vector_store.delete(doc_id)
        
        logger.info(f"Document deleted successfully, path: {file_path}")

    @abstractmethod
    def _do_delete(self, doc_id: str, file_path: Path):
        """执行具体的删除逻辑 - 子类必须实现此方法
        
        Args:
            doc_id: 文档唯一标识
            file_path: 文件存储路径
        """
        pass
    
    def clear(self):
        """清空数据库和所有向量存储"""
        for vector_store in self.vector_stores:
            vector_store.clear()
        if Path(self.database_path).exists():
            shutil.rmtree(self.database_path)
