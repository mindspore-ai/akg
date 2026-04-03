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

# 缓存配置管理

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CACHE_CONFIG: Dict[str, Any] = {
    "max_memory_size": 200,
    "cache_file_path": "~/.akg/llm_cache/llm_test_cache.json",
    "expire_seconds": -1,
    "auto_clean_expired": True,
}


def _get_env_override(key: str) -> str:
    """Read cache env override with AKG_AGENTS_* first, then AIKG_* fallback."""
    env_key_primary = f"AKG_AGENTS_{key}"
    env_key_compat = f"AIKG_{key}"
    return (os.getenv(env_key_primary) or os.getenv(env_key_compat) or "").strip()


def load_cache_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load cache config from YAML with safe defaults."""
    config = dict(DEFAULT_CACHE_CONFIG)

    env_config_path = _get_env_override("CACHE_CONFIG_PATH")
    if env_config_path:
        path = Path(env_config_path).expanduser()
    elif config_path:
        path = Path(config_path).expanduser()
    else:
        path = Path(__file__).resolve().parents[1] / "config" / "cache_config.yaml"

    if not path.exists():
        logger.info("Cache config not found; using defaults.")
        return config

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Failed to load cache config: {exc}")
        return config

    if isinstance(raw, dict) and "cache" in raw:
        raw = raw.get("cache") or {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in DEFAULT_CACHE_CONFIG:
                config[key] = value
            else:
                logger.debug(f"Ignoring unknown cache config key: {key}")
    else:
        logger.warning("Cache config format invalid; using defaults.")

    env_cache_file_path = _get_env_override("CACHE_FILE_PATH")
    if env_cache_file_path:
        config["cache_file_path"] = env_cache_file_path

    return config
