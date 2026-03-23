import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CACHE_CONFIG: Dict[str, Any] = {
    "enable": True,
    "max_memory_size": 200,
    "cache_file_path": "~/.akg/llm_cache/llm_test_cache.json5",
    "expire_seconds": -1,
    "auto_clean_expired": True,
}


def load_cache_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load cache config from YAML with safe defaults."""
    config = dict(DEFAULT_CACHE_CONFIG)

    if config_path:
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

    return config
