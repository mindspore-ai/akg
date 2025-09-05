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
import random
import string
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .common_utils import get_md5_hash

logger = logging.getLogger(__name__)

COLLECTOR_VERSION = "1.0.0"


def get_version() -> str:
    return COLLECTOR_VERSION


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
            # 生成高度随机的session_id，避免高并发或多服务器环境下的重复
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            uuid_part = str(uuid.uuid4())[:8]  # UUID的前8位
            random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))  # 6位随机字符
            process_id = os.getpid()  # 进程ID，用于区分不同进程
            cls._instance._session_id = f"session_{timestamp}_{uuid_part}_{random_part}_{process_id}"
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

    def _generate_filename(self, agent_name: str, hash_value: str, sequence_id: Optional[int] = None) -> str:
        """
        生成文件名，格式为 {_session_id}_{sequence_id}_{agent_name}_{hash}.json

        Args:
            agent_name: 代理名称
            hash_value: 原始哈希值
            sequence_id: 序列ID，如果提供则直接使用，如果不提供则使用当前_counter并递增

        Returns:
            str: 生成的文件名
        """
        safe_agent_name = agent_name.replace('/', '_').replace('\\', '_')
        safe_hash = hash_value.replace('/', '_').replace('\\', '_')
        safe_session_id = self._session_id.replace('/', '_').replace('\\', '_')

        # 如果没有提供sequence_id，则使用当前counter并递增（用于测试等场景）
        if sequence_id is None:
            sequence_id = self._counter
            self._counter += 1

        return f"{safe_session_id}_{sequence_id}_{safe_agent_name}_{safe_hash}.json"

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
                agent_name = data.get('agent_name', '')
                hash_value = data.get('hash', '')

                # 数据字段校验
                self._validate_data_fields(data)

                key = (agent_name, hash_value)

                entry = {
                    "version": COLLECTOR_VERSION,
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

    def _validate_data_fields(self, data: Dict[str, Any]) -> None:
        """
        校验数据字段完整性

        Args:
            data: 要校验的数据字典
        """
        agent_name = data.get('agent_name', '')

        if agent_name == "feature_extractor":
            # feature_extractor agent允许字段不全
            required_fields = ['hash', 'agent_name', 'model_name',
                               'content', 'formatted_prompt', 'reasoning_content',
                               'response_metadata']
        else:
            # 标准agent应该包含的必需字段
            required_fields = [
                'hash', 'agent_name', 'op_name', 'dsl', 'backend',
                'arch', 'framework', 'workflow_name', 'task_desc', 'model_name',
                'content', 'formatted_prompt', 'reasoning_content',
                'response_metadata'
            ]

        missing_fields = []
        empty_fields = []

        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif self._is_empty_value(data[field]):
                empty_fields.append(field)

        # 记录缺失或空值的字段
        if missing_fields:
            logger.warning(f"Agent '{agent_name}' 缺少必需字段: {missing_fields}")
        if empty_fields:
            logger.warning(f"Agent '{agent_name}' 字段为空: {empty_fields}")

    def _is_empty_value(self, value) -> bool:
        """
        判断值是否为空

        Args:
            value: 要判断的值

        Returns:
            bool: 是否为空值
        """
        if value is None:
            return True
        elif isinstance(value, str) and value.strip() == "":
            return True
        elif isinstance(value, (list, dict)) and len(value) == 0:
            return True
        return False

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
                    send_data = {
                        # 发送时的元数据
                        "send_timestamp": datetime.now().isoformat(),
                        "send_task_id": task_id if task_id else "",

                        # entry的核心属性
                        "version": entry.get("version"),
                        "session_id": entry.get("session_id"),
                        "sequence_id": entry.get("sequence_id"),

                        # 原始数据的一级属性
                        **entry.get("data", {})
                    }

                    # 生成文件名并保存（使用数据收集时的sequence_id作为前缀）
                    filename = self._generate_filename(agent_name, hash_value, entry.get("sequence_id", None))
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
                "version": COLLECTOR_VERSION,
                "hash": "database",
                "backend": task_info.get("backend", ""),
                "arch": task_info.get("arch_name", ""),
                "framework": task_info.get("framework", ""),
                "dsl": task_info.get("dsl", ""),
                "workflow_name": task_info.get("workflow_name", ""),
                "framework_code": task_info.get("task_desc", ""),
                "impl_code": task_info.get("coder_code", ""),
                "profile": task_info.get("profile_res", ())  # 保存完整的三元组
            }

            # 生成文件名并保存
            filename = self._generate_filename("database", "database")
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
