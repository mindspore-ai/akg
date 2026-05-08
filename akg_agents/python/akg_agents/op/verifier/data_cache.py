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
import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_VERIFIER_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".akg", "verifier_data_cache")
DEFAULT_CACHE_LOCK_TIMEOUT_SECONDS = 600
DEFAULT_CACHE_LOCK_STALE_SECONDS = 3600


@dataclass
class VerifierDataCacheConfig:
    enabled: bool = False
    cache_dir: str = DEFAULT_VERIFIER_CACHE_DIR
    cache_reference_data: bool = True
    cache_baseline_result: bool = True


def _get_data_cache_options(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    raw = config.get("data_cache") or {}
    if not isinstance(raw, dict):
        return {}
    return raw


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


def _normalize_cache_dir(cache_dir: Any) -> str:
    raw = str(cache_dir or DEFAULT_VERIFIER_CACHE_DIR).strip() or DEFAULT_VERIFIER_CACHE_DIR
    expanded = os.path.expandvars(os.path.expanduser(raw))
    return str(Path(expanded))


def load_verifier_data_cache_config(config: Optional[Dict[str, Any]] = None) -> VerifierDataCacheConfig:
    raw = dict(_get_data_cache_options(config))

    env_enabled = _get_env_override("VERIFY_DATA_CACHE")
    env_cache_dir = _get_env_override("VERIFY_DATA_CACHE_DIR")

    enabled = _parse_bool(raw.get("enabled"), False)
    if env_enabled:
        enabled = _parse_bool(env_enabled, enabled)

    cache_dir = str(raw.get("cache_dir") or DEFAULT_VERIFIER_CACHE_DIR)
    if env_cache_dir:
        cache_dir = env_cache_dir
    cache_dir = _normalize_cache_dir(cache_dir)

    cache_reference_data = _parse_bool(raw.get("cache_reference_data"), True)
    cache_baseline_result = _parse_bool(raw.get("cache_baseline_result"), True)

    return VerifierDataCacheConfig(
        enabled=enabled,
        cache_dir=cache_dir,
        cache_reference_data=cache_reference_data,
        cache_baseline_result=cache_baseline_result,
    )


def get_verifier_data_cache_key_id(config: Optional[Dict[str, Any]], default_task_id: str = "") -> str:
    raw = _get_data_cache_options(config)
    cache_key_id = str(raw.get("cache_key_id") or "").strip()
    if cache_key_id:
        return cache_key_id
    return str(default_task_id or "").strip()


def set_verifier_data_cache_key_id(
    config: Optional[Dict[str, Any]],
    cache_key_id: str,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(config, dict):
        return
    raw = config.get("data_cache")
    if not isinstance(raw, dict):
        raw = {}
        config["data_cache"] = raw
    if overwrite or not str(raw.get("cache_key_id") or "").strip():
        raw["cache_key_id"] = str(cache_key_id or "").strip()


def build_workflow_data_cache_key_id(
    *,
    op_name: str,
    framework: str,
    dsl: str,
    backend: str,
    arch: str,
    bench_type: str,
) -> str:
    return ":".join(
        str(component or "").strip()
        for component in (op_name, framework, dsl, backend, arch, bench_type)
    )


def set_workflow_data_cache_key_id(
    config: Optional[Dict[str, Any]],
    *,
    op_name: str,
    framework: str,
    dsl: str,
    backend: str,
    arch: str,
    bench_type: str,
    overwrite: bool = False,
) -> None:
    set_verifier_data_cache_key_id(
        config,
        build_workflow_data_cache_key_id(
            op_name=op_name,
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            bench_type=bench_type,
        ),
        overwrite=overwrite,
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
    task_id: str = "",
) -> str:
    payload = {
        "type": "reference_data",
        "version": 2,
        "op_name": op_name,
        "task_id": task_id,
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
    task_id: str = "",
) -> str:
    payload = {
        "type": "baseline_profile",
        "version": 2,
        "op_name": op_name,
        "task_id": task_id,
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


def _cache_root(cfg: VerifierDataCacheConfig) -> Path:
    return Path(_normalize_cache_dir(cfg.cache_dir))


def _cache_stem(op_name: str, cache_key: str) -> str:
    return f"{_sanitize_name(op_name)}_{cache_key}"


def _reference_cache_paths(
    cfg: VerifierDataCacheConfig,
    op_name: str,
    cache_key: str,
) -> tuple[Path, Path]:
    base_dir = _cache_root(cfg) / "reference"
    stem = _cache_stem(op_name, cache_key)
    return base_dir / f"{stem}.pt", base_dir / f"{stem}.json"


def _baseline_cache_path(
    cfg: VerifierDataCacheConfig,
    op_name: str,
    cache_key: str,
) -> Path:
    base_dir = _cache_root(cfg) / "baseline"
    stem = _cache_stem(op_name, cache_key)
    return base_dir / f"{stem}.json"


def get_reference_cache_file_path(
    cfg: VerifierDataCacheConfig,
    *,
    op_name: str,
    cache_key: str,
) -> str:
    data_path, _ = _reference_cache_paths(cfg, op_name, cache_key)
    return str(data_path)


def get_baseline_cache_file_path(
    cfg: VerifierDataCacheConfig,
    *,
    op_name: str,
    cache_key: str,
) -> str:
    return str(_baseline_cache_path(cfg, op_name, cache_key))


def build_sol_problem_cache_identity(sol_problem_dir: str) -> str:
    """Build a stable cache identity from the default SOL case directory."""
    problem_path = Path(os.path.expandvars(os.path.expanduser(str(sol_problem_dir or ""))))
    if not problem_path.is_dir():
        raise FileNotFoundError(f"SOL problem directory not found: {problem_path}")

    definition_path = problem_path / "definition.json"
    workload_path = problem_path / "workload.jsonl"
    reference_path = problem_path / "reference.py"
    for path in (definition_path, workload_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required SOL file: {path}")

    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    workloads = [
        json.loads(line)
        for line in workload_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference_code = reference_path.read_text(encoding="utf-8")
    payload = {
        "definition": definition,
        "workloads": workloads,
        "reference_code": _normalize_framework_code(reference_code),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cache_lock_dir(
    cfg: VerifierDataCacheConfig,
    *,
    namespace: str,
    op_name: str,
    cache_key: str,
) -> Path:
    safe_namespace = _sanitize_name(namespace)
    return _cache_root(cfg) / ".locks" / f"{safe_namespace}_{_cache_stem(op_name, cache_key)}.lock"


def _is_stale_lock(lock_dir: Path, stale_after_seconds: float) -> bool:
    if stale_after_seconds <= 0:
        return True
    try:
        return (time.time() - lock_dir.stat().st_mtime) > stale_after_seconds
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _try_acquire_cache_lock(lock_dir: Path) -> bool:
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir()
    except FileExistsError:
        return False
    owner_payload = {
        "pid": os.getpid(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (lock_dir / "owner.json").write_text(json.dumps(owner_payload), encoding="utf-8")
    except OSError:
        pass
    return True


def _release_cache_lock(lock_dir: Path) -> None:
    try:
        shutil.rmtree(lock_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(f"Failed to release verifier data cache lock {lock_dir}: {exc}")


@asynccontextmanager
async def verifier_data_cache_lock(
    cfg: VerifierDataCacheConfig,
    *,
    namespace: str,
    op_name: str,
    cache_key: str,
    timeout_seconds: float = DEFAULT_CACHE_LOCK_TIMEOUT_SECONDS,
    stale_after_seconds: float = DEFAULT_CACHE_LOCK_STALE_SECONDS,
):
    """Serialize cache mutations for one cache entry across processes."""
    lock_dir = _cache_lock_dir(cfg, namespace=namespace, op_name=op_name, cache_key=cache_key)
    start = time.monotonic()
    acquired = False
    try:
        while True:
            if _try_acquire_cache_lock(lock_dir):
                acquired = True
                break
            if _is_stale_lock(lock_dir, stale_after_seconds):
                _release_cache_lock(lock_dir)
                continue
            if timeout_seconds <= 0 or (time.monotonic() - start) >= timeout_seconds:
                raise TimeoutError(f"Timed out waiting for verifier data cache lock: {lock_dir}")
            await asyncio.sleep(0.2)

        yield
    finally:
        if acquired:
            _release_cache_lock(lock_dir)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        logger.debug(f"Failed to open verifier cache directory for fsync {path}: {exc}")
        return
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        logger.debug(f"Failed to fsync verifier cache directory {path}: {exc}")
    finally:
        os.close(dir_fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


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
