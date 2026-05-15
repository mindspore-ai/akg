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

import hashlib
import json
import os
import logging
import re
import time
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_legacy_json5(data_str: str) -> str:
    """Best-effort migration from simple JSON5-like content to valid JSON."""
    normalized = re.sub(r"/\*.*?\*/", "", data_str, flags=re.DOTALL)
    normalized = re.sub(r"//.*?$", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    return normalized


def _content_needs_migration(data_str: str) -> bool:
    return bool(re.search(r"/\*|//|,\s*[}\]]", data_str))


@contextmanager
def _file_lock(lock_path: Path, timeout_seconds: float = 5.0, retry_interval: float = 0.05):
    """Cross-platform lock using an atomic lock file create/delete cycle."""
    fd = None
    start = time.monotonic()

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if time.monotonic() - start >= timeout_seconds:
                raise TimeoutError(f"Acquire lock timeout: {lock_path}")
            time.sleep(retry_interval)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def generate_cache_key(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
    **kwargs
) -> str:
    key_payload = {
        "messages": messages,
        "tools": tools if tools is not None else [],
        "params": kwargs
    }
    sorted_json_str = json.dumps(key_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    md5_hash = hashlib.md5(sorted_json_str.encode("utf-8")).hexdigest()
    logger.debug(f"Generated cache key: {md5_hash}")
    return md5_hash


def serialize_cache_data(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def deserialize_cache_data(data_str: str) -> Dict[str, Any]:
    try:
        return json.loads(data_str)
    except Exception as e:
        try:
            return json.loads(_normalize_legacy_json5(data_str))
        except Exception:
            logger.error(f"Cache data deserialize failed: {e}")
            return {}


def init_cache_file(file_path: str) -> Path:
    cache_path = Path(file_path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        cache_path.write_text("{}", encoding="utf-8")
        logger.info(f"Created cache file: {cache_path}")
    return cache_path


def read_cache_file(file_path: str) -> Dict[str, Any]:
    cache_path = init_cache_file(file_path)
    try:
        content = cache_path.read_text(encoding="utf-8")
        data = deserialize_cache_data(content)
        if data and _content_needs_migration(content):
            cache_path.write_text(serialize_cache_data(data), encoding="utf-8")
        return data
    except Exception as e:
        logger.error(f"Read cache file failed: {e}")
        return {}


def write_cache_file(file_path: str, cache_data: Dict[str, Any]) -> None:
    cache_path = init_cache_file(file_path)
    lock_path = Path(str(cache_path) + ".lock")
    temp_path = Path(str(cache_path) + ".tmp")
    try:
        with _file_lock(lock_path):
            content = serialize_cache_data(cache_data)
            temp_path.write_text(content, encoding="utf-8")
            os.replace(temp_path, cache_path)
        logger.debug("Cache file written successfully")
    except Exception as e:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        logger.error(f"Write cache file failed: {e}")
