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

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .common_utils import get_md5_hash

logger = logging.getLogger(__name__)


class Collector:
    """异步单例数据收集器"""
    _instance = None
    _async_lock = None
    _initialized = False

    def __new__(cls):
        """确保单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._store = {}
            cls._instance._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cls._instance._counter = 0
            cls._instance._config = None
            cls._instance._save_dir = None
        return cls._instance

    def set_config(self, config: Optional[dict]):
        """
        设置配置，主要用于获取log_dir

        Args:
            config: 配置字典，包含log_dir等信息
        """
        self._config = config
        if config and 'log_dir' in config:
            # 创建save_data目录
            log_dir = Path(os.path.expanduser(config['log_dir'])).parent
            self._save_dir = log_dir / "save_data"
            self._save_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created save directory: {self._save_dir}")
        else:
            # 默认目录
            self._save_dir = Path.cwd() / "save_data"
            self._save_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"No log_dir in config, using default: {self._save_dir}")

    async def _ensure_initialized(self):
        """确保异步组件已初始化"""
        if not self._initialized:
            if self._async_lock is None:
                self._async_lock = asyncio.Lock()
            self._initialized = True

    def _generate_filename(self, agent_name: str, hash_value: str, data: Optional[dict] = None) -> str:
        """
        生成文件名，格式为 {agent_name}_{hash}_{时间戳}_{内容哈希}.json

        Args:
            agent_name: 代理名称
            hash_value: 原始哈希值
            data: 数据内容，用于生成内容哈希

        Returns:
            str: 生成的文件名
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 精确到毫秒
        safe_agent_name = agent_name.replace('/', '_').replace('\\', '_')
        safe_hash = hash_value.replace('/', '_').replace('\\', '_')

        # 生成基于数据内容的哈希值
        content_hash = ""
        if data:
            try:
                content_hash = get_md5_hash(**data)[:8]
            except Exception as e:
                logger.warning(f"Failed to generate content hash: {e}")
                # 如果无法生成内容哈希，使用时间戳的纳秒部分
                content_hash = str(datetime.now().timestamp()).replace('.', '')[-8:]
        else:
            # 没有数据时使用时间戳的纳秒部分
            content_hash = str(datetime.now().timestamp()).replace('.', '')[-8:]

        return f"{safe_agent_name}_{safe_hash}_{timestamp}_{content_hash}.json"

    def _save_json_file(self, data: dict, filename: str) -> bool:
        """
        保存JSON文件

        Args:
            data: 要保存的数据
            filename: 文件名

        Returns:
            bool: 保存是否成功
        """
        try:
            if self._save_dir is None:
                logger.error("Save directory not initialized, call set_config first")
                return False

            file_path = self._save_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, separators=(',', ': '))

            logger.debug(f"Saved JSON file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON file {filename}: {e}")
            return False

    async def collect(self, data: Dict[str, Any]):
        """
        收集数据，自动覆盖相同(agent_name, hash)的数据

        Args:
            data: 要收集的数据字典
        """
        await self._ensure_initialized()

        try:
            async with self._async_lock:
                agent_name = data.get('agent_name', 'unknown')
                hash_value = data.get('hash', 'unknown')

                key = (agent_name, hash_value)

                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "loop_time": asyncio.get_event_loop().time(),
                    "session_id": self._session_id,
                    "sequence_id": self._counter,
                    "data": data,
                }

                was_overwrite = key in self._store
                self._store[key] = entry
                self._counter += 1

                if was_overwrite:
                    logger.debug(f"Overwriting data for key {key}")
                else:
                    logger.debug(f"Collected new data item {self._counter}: {agent_name}")

        except Exception as e:
            logger.warning(f"Failed to collect data: {e}")

    async def prepare_and_remove_data(self, task_id: str = "") -> List[str]:
        """
        整理要发送的数据并从collector中移除，保存为JSON文件

        Args:
            task_id: 任务ID，如果提供则准备该任务的数据 + 所有无task_id的数据
                    如果不提供则只准备所有无task_id的数据

        Returns:
            List[str]: 保存的文件路径列表
        """
        await self._ensure_initialized()

        async with self._async_lock:
            data_to_prepare = []
            keys_to_remove = []

            # 如果提供了task_id，准备该任务的数据
            if task_id:
                for key, entry in self._store.items():
                    entry_data = entry.get("data", {})
                    entry_task_id = entry_data.get("task_id")

                    # 收集指定task_id的数据
                    if entry_task_id == task_id:
                        data_to_prepare.append((key, entry))
                        keys_to_remove.append(key)

            # 总是收集无task_id的数据
            for key, entry in self._store.items():
                entry_data = entry.get("data", {})
                entry_task_id = entry_data.get("task_id")

                # 收集无task_id的数据（避免重复添加）
                if (entry_task_id is None or entry_task_id == "") and key not in keys_to_remove:
                    data_to_prepare.append((key, entry))
                    keys_to_remove.append(key)

            # 从collector中移除这些数据（原子操作）
            for key in keys_to_remove:
                del self._store[key]
                logger.debug(f"Removed data from collector: {key}")

            # 保存JSON文件
            saved_files = []
            for (agent_name, hash_value), entry in data_to_prepare:
                try:
                    # 构建一级JSON结构
                    send_data = {
                        # 发送时的元数据
                        "send_timestamp": datetime.now().isoformat(),
                        "send_task_id": task_id if task_id else "",

                        # entry的一级属性
                        "collect_timestamp": entry.get("timestamp"),
                        "collect_loop_time": entry.get("loop_time"),
                        "collect_session_id": entry.get("session_id"),
                        "collect_sequence_id": entry.get("sequence_id"),

                        # 原始数据的一级属性
                        **entry.get("data", {})
                    }

                    # 生成文件名并保存
                    filename = self._generate_filename(agent_name, hash_value, entry.get("data"))
                    if self._save_json_file(send_data, filename):
                        saved_files.append(str(self._save_dir / filename))

                except Exception as e:
                    logger.error(f"Failed to save entry to JSON file: {e}")
                    continue

            logger.debug(f"Saved {len(saved_files)} JSON files for (task_id: {task_id or 'none'})")
            return saved_files

    def prepare_database_data(self, task_info: dict) -> str:
        """
        根据task_info准备database所需的数据并保存为JSON文件

        Args:
            task_info: conductor的task_info字典，包含所有生成的代码和结果

        Returns:
            str: 保存的数据库JSON文件路径，如果保存失败返回空字符串
        """
        try:
            database_data = {
                "hash": "database",
                "backend": task_info.get("backend", ""),
                "arch": task_info.get("arch_name", ""),  # base_doc中使用arch_name
                "framework": task_info.get("framework", ""),
                "dsl": task_info.get("dsl", ""),
                "framework_code": task_info.get("task_desc", ""),
                "impl_code": task_info.get("coder_code", ""),
                "profile": task_info.get("profile_res", ())  # 保存完整的三元组
            }

            # 生成文件名并保存
            filename = self._generate_filename("database", "database", database_data)
            if self._save_json_file(database_data, filename):
                file_path = str(self._save_dir / filename)
                logger.debug(f"Saved database JSON file: {file_path}")
                return file_path
            else:
                logger.error("Failed to save database JSON file")
                return ""

        except Exception as e:
            logger.warning(f"Failed to prepare database data: {e}")
            return ""


# 全局收集器实例
_global_collector = None


async def get_collector() -> Collector:
    """
    获取全局collector实例

    Returns:
        Collector实例
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = Collector()
    return _global_collector
