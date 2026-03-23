import hashlib
import json5
import os
import fcntl
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


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
    sorted_json_str = json5.dumps(key_payload, sort_keys=True, ensure_ascii=False)
    md5_hash = hashlib.md5(sorted_json_str.encode("utf-8")).hexdigest()
    logger.debug(f"Generated cache key: {md5_hash}")
    return md5_hash


def serialize_cache_data(data: Dict[str, Any]) -> str:
    return json5.dumps(data, ensure_ascii=False, indent=2)


def deserialize_cache_data(data_str: str) -> Dict[str, Any]:
    try:
        return json5.loads(data_str)
    except Exception as e:
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
        return deserialize_cache_data(content)
    except Exception as e:
        logger.error(f"Read cache file failed: {e}")
        return {}


def write_cache_file(file_path: str, cache_data: Dict[str, Any]) -> None:
    cache_path = init_cache_file(file_path)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            content = serialize_cache_data(cache_data)
            f.write(content)
            fcntl.flock(f, fcntl.LOCK_UN)
        logger.debug("Cache file written successfully")
    except Exception as e:
        logger.error(f"Write cache file failed: {e}")