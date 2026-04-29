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

"""Persistent data cache helpers for KernelVerifier."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_VERIFIER_CACHE_DIR = "~/.akg/verifier_data_cache"


@dataclass
class VerifierDataCacheConfig:
    enabled: bool = False
    cache_dir: str = DEFAULT_VERIFIER_CACHE_DIR
    cache_reference_data: bool = True
    cache_baseline_result: bool = True


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _get_env_override(key: str) -> str:
    primary = f"AKG_AGENTS_{key}"
    compat = f"AIKG_{key}"
    return (os.getenv(primary) or os.getenv(compat) or "").strip()


def load_verifier_data_cache_config(config: Optional[Dict[str, Any]] = None) -> VerifierDataCacheConfig:
    raw = dict((config or {}).get("data_cache") or {})

    env_enabled = _get_env_override("VERIFY_DATA_CACHE")
    env_cache_dir = _get_env_override("VERIFY_DATA_CACHE_DIR")

    enabled = _parse_bool(raw.get("enabled"), False)
    if env_enabled:
        enabled = _parse_bool(env_enabled, enabled)

    cache_dir = str(raw.get("cache_dir") or DEFAULT_VERIFIER_CACHE_DIR)
    if env_cache_dir:
        cache_dir = env_cache_dir

    cache_reference_data = _parse_bool(raw.get("cache_reference_data"), True)
    cache_baseline_result = _parse_bool(raw.get("cache_baseline_result"), True)

    return VerifierDataCacheConfig(
        enabled=enabled,
        cache_dir=cache_dir,
        cache_reference_data=cache_reference_data,
        cache_baseline_result=cache_baseline_result,
    )


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_") or "entry"


def _normalize_framework_code(code: str) -> str:
    stripped = (code or "").strip()
    if not stripped:
        return ""
    try:
        tree = ast.parse(stripped)
        return ast.dump(tree, include_attributes=False)
    except SyntaxError:
        return stripped


def _build_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_reference_cache_key(
    *,
    op_name: str,
    framework_code: str,
    framework: str,
    backend: str,
    arch: str,
    bench_type: str,
) -> str:
    payload = {
        "type": "reference_data",
        "version": 1,
        "op_name": op_name,
        "framework": framework,
        "backend": backend,
        "arch": arch,
        "bench_type": bench_type,
        "framework_code": _normalize_framework_code(framework_code),
    }
    return _build_hash(payload)


def build_baseline_cache_key(
    *,
    op_name: str,
    framework_code: str,
    framework: str,
    backend: str,
    arch: str,
    bench_type: str,
    warmup_times: int,
    run_times: int,
    dsl: str = "",
) -> str:
    payload = {
        "type": "baseline_profile",
        "version": 1,
        "op_name": op_name,
        "framework": framework,
        "dsl": dsl,
        "backend": backend,
        "arch": arch,
        "bench_type": bench_type,
        "warmup_times": int(warmup_times),
        "run_times": int(run_times),
        "framework_code": _normalize_framework_code(framework_code),
    }
    return _build_hash(payload)


def _reference_cache_paths(
    cfg: VerifierDataCacheConfig,
    op_name: str,
    cache_key: str,
) -> tuple[Path, Path]:
    base_dir = Path(cfg.cache_dir).expanduser() / "reference"
    stem = f"{_sanitize_name(op_name)}_{cache_key}"
    return base_dir / f"{stem}.pt", base_dir / f"{stem}.json"


def _baseline_cache_path(
    cfg: VerifierDataCacheConfig,
    op_name: str,
    cache_key: str,
) -> Path:
    base_dir = Path(cfg.cache_dir).expanduser() / "baseline"
    stem = f"{_sanitize_name(op_name)}_{cache_key}"
    return base_dir / f"{stem}.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def read_reference_data_from_cache(
    cfg: VerifierDataCacheConfig,
    *,
    op_name: str,
    cache_key: str,
) -> Optional[bytes]:
    if not cfg.enabled or not cfg.cache_reference_data:
        return None
    data_path, _ = _reference_cache_paths(cfg, op_name, cache_key)
    if not data_path.exists():
        return None
    try:
        return data_path.read_bytes()
    except Exception as exc:
        logger.warning(f"Failed to read verifier reference cache {data_path}: {exc}")
        return None


def write_reference_data_to_cache(
    cfg: VerifierDataCacheConfig,
    *,
    op_name: str,
    cache_key: str,
    reference_data: bytes,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    if not cfg.enabled or not cfg.cache_reference_data or not reference_data:
        return None
    data_path, meta_path = _reference_cache_paths(cfg, op_name, cache_key)
    payload = {
        "type": "reference_data",
        "cache_key": cache_key,
        "op_name": op_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": len(reference_data),
    }
    if metadata:
        payload.update(metadata)
    try:
        _atomic_write_bytes(data_path, reference_data)
        _atomic_write_json(meta_path, payload)
        return data_path
    except Exception as exc:
        logger.warning(f"Failed to write verifier reference cache {data_path}: {exc}")
        return None


def delete_reference_data_from_cache(
    cfg: VerifierDataCacheConfig,
    *,
    op_name: str,
    cache_key: str,
) -> None:
    data_path, meta_path = _reference_cache_paths(cfg, op_name, cache_key)
    for path in (data_path, meta_path):
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"Failed to delete verifier reference cache {path}: {exc}")


def read_baseline_result_from_cache(
    cfg: VerifierDataCacheConfig,
    *,
    op_name: str,
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    if not cfg.enabled or not cfg.cache_baseline_result:
        return None
    result_path = _baseline_cache_path(cfg, op_name, cache_key)
    if not result_path.exists():
        return None
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to read verifier baseline cache {result_path}: {exc}")
        try:
            result_path.unlink(missing_ok=True)
        except Exception as delete_exc:
            logger.warning(f"Failed to delete corrupted verifier baseline cache {result_path}: {delete_exc}")
        return None


def write_baseline_result_to_cache(
    cfg: VerifierDataCacheConfig,
    *,
    op_name: str,
    cache_key: str,
    result_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    if not cfg.enabled or not cfg.cache_baseline_result or not result_data:
        return None
    result_path = _baseline_cache_path(cfg, op_name, cache_key)
    payload = {
        "type": "baseline_profile",
        "cache_key": cache_key,
        "op_name": op_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **result_data,
    }
    if metadata:
        payload.update(metadata)
    try:
        _atomic_write_json(result_path, payload)
        return result_path
    except Exception as exc:
        logger.warning(f"Failed to write verifier baseline cache {result_path}: {exc}")
        return None


def delete_baseline_result_from_cache(
    cfg: VerifierDataCacheConfig,
    *,
    op_name: str,
    cache_key: str,
) -> None:
    result_path = _baseline_cache_path(cfg, op_name, cache_key)
    try:
        result_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Failed to delete verifier baseline cache {result_path}: {exc}")


def extract_baseline_time_us(result_data: Optional[Dict[str, Any]]) -> Optional[float]:
    if not result_data:
        return None
    for key in ("avg_time_us", "execution_time_us", "base_time_us"):
        value = result_data.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0 and numeric < float("inf"):
            return numeric
    return None


def build_baseline_cache_payload(
    *,
    base_time_us: float,
    warmup_times: int,
    run_times: int,
    method: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "avg_time_us": base_time_us,
        "execution_time_us": base_time_us,
        "execution_time_ms": base_time_us / 1000.0,
        "warmup_times": int(warmup_times),
        "run_times": int(run_times),
    }
    if method:
        payload["method"] = method
    if extra:
        payload.update(extra)
    return payload


def cache_config_to_dict(cfg: VerifierDataCacheConfig) -> Dict[str, Any]:
    return asdict(cfg)
