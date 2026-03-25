# Copyright 2026 Huawei Technologies Co., Ltd
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

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from cachetools import LRUCache

from .cache_utils import (
    generate_cache_key,
    read_cache_file,
    write_cache_file,
    serialize_cache_data
)

logger = logging.getLogger(__name__)


class LLMCache:
    def __init__(
        self,
        max_memory_size: int = 200,
        cache_file_path: str = "~/.akg/llm_cache/llm_test_cache.json",
        expire_seconds: int = -1,
        auto_clean_expired: bool = True,
        enable: bool = True
    ):
        self.enable = enable
        self.max_memory_size = max_memory_size
        self.cache_file_path = cache_file_path
        self.expire_seconds = expire_seconds
        self.auto_clean_expired = auto_clean_expired

        self._memory_cache = LRUCache(maxsize=self.max_memory_size)
        self._local_cache = self._load_local_cache()

        if self.auto_clean_expired:
            self.clean_expired_cache()

        logger.info(
            f"LLMCache initialized: enable={self.enable}, "
            f"memory_size={max_memory_size}, local_cache_path={cache_file_path}"
        )

    def _load_local_cache(self) -> Dict[str, Any]:
        if not self.enable:
            return {}
        return read_cache_file(self.cache_file_path)

    def _save_local_cache(self) -> None:
        if not self.enable:
            return
        write_cache_file(self.cache_file_path, self._local_cache)

    def _is_cache_expired(self, cache_item: Dict[str, Any]) -> bool:
        if self.expire_seconds == -1:
            return False
        create_time = cache_item.get("create_time", 0)
        return time.time() - create_time > self.expire_seconds

    def _read_valid_cache_item(self, cache_store: Dict[str, Any], cache_key: str, source_name: str) -> Optional[Dict[str, Any]]:
        cache_item = cache_store.get(cache_key)
        if cache_item is None:
            return None

        if self._is_cache_expired(cache_item):
            del cache_store[cache_key]
            return None

        logger.info(f"✅ {source_name} cache hit, key={cache_key}")
        return cache_item

    def get(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        if not self.enable:
            return None

        cache_key = generate_cache_key(messages, tools, **kwargs)

        memory_item = self._read_valid_cache_item(self._memory_cache, cache_key, "Memory")
        if memory_item is not None:
            return memory_item.get("result")

        local_size_before = len(self._local_cache)
        local_item = self._read_valid_cache_item(self._local_cache, cache_key, "Local")
        if local_item is not None:
            self._memory_cache[cache_key] = local_item
            return local_item.get("result")

        if len(self._local_cache) != local_size_before:
            self._save_local_cache()

        logger.info(f"❌ Cache miss, key={cache_key}")
        return None

    def set(
        self,
        messages: List[Dict[str, str]],
        result: Dict[str, Any],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> None:
        if not self.enable:
            return

        cache_key = generate_cache_key(messages, tools, **kwargs)

        cache_item = {
            "cache_key": cache_key,
            "messages": messages,
            "result": result,
            "tools": tools if tools is not None else [],
            "params": kwargs,
            "create_time": time.time(),
            "expire_seconds": self.expire_seconds
        }

        self._memory_cache[cache_key] = cache_item
        self._local_cache[cache_key] = cache_item
        self._save_local_cache()

        logger.debug(f"Cache stored successfully, key={cache_key}")

    def clean_expired_cache(self) -> int:
        expired_keys = []
        for key, item in self._local_cache.items():
            if self._is_cache_expired(item):
                expired_keys.append(key)

        for key in expired_keys:
            del self._local_cache[key]
            if key in self._memory_cache:
                del self._memory_cache[key]

        if expired_keys:
            self._save_local_cache()
            logger.info(f"Cleaned {len(expired_keys)} expired cache items")

        return len(expired_keys)

    def clear_all_cache(self) -> None:
        self._memory_cache.clear()
        self._local_cache.clear()
        self._save_local_cache()
        logger.warning("All cache cleared")

    def get_all_cache_keys(self) -> List[str]:
        return list(self._local_cache.keys())

    def get_cache_by_key(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return self._local_cache.get(cache_key)

    def get_cache_by_messages(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        cache_key = generate_cache_key(messages, tools, **kwargs)
        return self.get_cache_by_key(cache_key)

    def export_all_cache(self, export_path: str) -> None:
        export_path = Path(export_path).expanduser()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(serialize_cache_data(self._local_cache), encoding="utf-8")
        logger.info(f"All cache exported to {export_path}")