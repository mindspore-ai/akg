#!/usr/bin/env python3
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

"""Shared helpers for benchmark_lite runners.

This module provides shared functionality for the benchmark_lite runners,
including backend configuration, device resolution, case discovery, task building,
result aggregation, performance evaluation, scoring, and output generation.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import queue
import shutil
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict


RUNNER_VERSION = "benchmark-lite-v2"
DEFAULT_TIERS = ("t1", "t2", "t3")


class BackendConfig(TypedDict):
    """Type definition for backend configuration."""
    name: str
    backend: str
    dsl: str
    arch: str
    skip_npu: bool


BACKEND_CONFIGS: Dict[str, BackendConfig] = {
    "cpu": {
        "name": "CPU",
        "backend": "cpu",
        "dsl": "cpp",
        "arch": "x86_64",
        "skip_npu": True,
    },
    "gpu": {
        "name": "GPU (CUDA)",
        "backend": "cuda",
        "dsl": "triton_cuda",
        "arch": "rtx3090",
        "skip_npu": True,
    },
    "npu": {
        "name": "NPU (Ascend)",
        "backend": "ascend",
        "dsl": "triton_ascend",
        "arch": "ascend910b4",
        "skip_npu": False,
    },
}


@dataclass(frozen=True)
class BenchLiteCase:
    tier: str
    case_name: str
    file_path: Path
    requires_npu: bool = False
    skip_reason: Optional[str] = None


@dataclass(frozen=True)
class DiscoveryResult:
    cases: List[BenchLiteCase]
    skipped_cases: List[BenchLiteCase]


def resolve_backend_config(
    backend: str,
    arch_override: Optional[str] = None,
    dsl_override: Optional[str] = None,
    backend_override: Optional[str] = None,
) -> BackendConfig:
    """
    Return the effective backend configuration.

    Args:
        backend: The backend name ('cpu', 'gpu', or 'npu').
        arch_override: Optional override for the architecture.
        dsl_override: Optional override for the DSL (e.g. 'pytorch', 'triton_cuda').
        backend_override: Optional override for the backend mapping name
            (e.g. 'cuda', 'ascend').  This is the value passed to
            ``register_local_worker`` and ``load_config``, not the CLI
            ``--backend`` selector.

    Returns:
        BackendConfig dictionary with effective settings.

    Raises:
        ValueError: If the backend name is unknown.
    """
    if backend not in BACKEND_CONFIGS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {list(BACKEND_CONFIGS.keys())}")

    config = dict(BACKEND_CONFIGS[backend])
    if arch_override:
        config["arch"] = arch_override
    if dsl_override:
        config["dsl"] = dsl_override
    if backend_override:
        config["backend"] = backend_override
    return config


def _parse_devices_env(raw_value: Optional[str]) -> List[int]:
    """
    Parse device IDs from environment variable string.

    Args:
        raw_value: Environment variable value (e.g., '0,1,2' or 'all').

    Returns:
        List of device IDs, empty list for 'all' or invalid input.
    """
    if not raw_value:
        return []

    stripped = raw_value.strip()
    if stripped.lower() == "all":
        return []

    devices: List[int] = []
    for item in stripped.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            device_id = int(item)
            if device_id >= 0:  # Validate non-negative device IDs
                devices.append(device_id)
        except ValueError:
            return []
    return devices


def _detect_visible_devices(backend: str) -> List[int]:
    if backend == "cpu":
        return [0]

    if backend == "gpu":
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except Exception:
            pass
        # Fallback when torch is unavailable: derive logical indices from env vars
        for env_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
            devices = _parse_devices_env(os.getenv(env_name))
            if devices:
                return list(range(len(devices)))
        return [0]

    if backend == "npu":
        try:
            import torch  # type: ignore

            npu_mod = getattr(torch, "npu", None)
            if npu_mod is not None and hasattr(npu_mod, "device_count"):
                count = npu_mod.device_count()
                if count:
                    return list(range(count))
        except Exception:
            pass
        # Fallback when torch is unavailable: derive logical indices from env vars
        for env_name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
            devices = _parse_devices_env(os.getenv(env_name))
            if devices:
                return list(range(len(devices)))
        return [0]

    return [0]


def _validate_devices(backend: str, devices: List[int]) -> List[int]:
    """Filter device list to only those actually available, with warnings."""
    if backend == "cpu":
        return devices

    try:
        import torch  # type: ignore
    except ImportError:
        return devices

    if backend == "gpu":
        if not torch.cuda.is_available():
            print(f"[WARN] CUDA is not available, cannot validate GPU devices")
            return devices
        total = torch.cuda.device_count()
        valid = []
        for d in devices:
            if d < total:
                valid.append(d)
            else:
                print(f"[WARN] GPU device {d} not available (only {total} device(s) visible), skipping")
        if not valid:
            raise RuntimeError(
                f"No available GPU devices among requested {devices} "
                f"(only {total} device(s) visible)"
            )
        return valid

    if backend == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None or not hasattr(npu_mod, "is_available") or not npu_mod.is_available():
            print(f"[WARN] NPU is not available, cannot validate NPU devices")
            return devices
        total = npu_mod.device_count()
        valid = []
        for d in devices:
            if d < total:
                valid.append(d)
            else:
                print(f"[WARN] NPU device {d} not available (only {total} device(s) visible), skipping")
        if not valid:
            raise RuntimeError(
                f"No available NPU devices among requested {devices} "
                f"(only {total} device(s) visible)"
            )
        return valid

    return devices


def resolve_devices(backend: str, devices_override: Optional[Iterable[int]] = None) -> List[int]:
    """
    Return effective device ids for a backend.

    Args:
        backend: The backend name ('cpu', 'gpu', or 'npu').
        devices_override: Optional iterable of device IDs.

    Returns:
        List of validated, available device IDs for the backend.

    Raises:
        RuntimeError: If all requested devices are unavailable.

    Note:
        Empty iterable triggers auto-detection, not use of empty list.
    """
    if devices_override is not None:
        devices_list = list(devices_override)
        if not devices_list:
            return _validate_devices(backend, _detect_visible_devices(backend))
        return _validate_devices(backend, [int(device) for device in devices_list])
    return _validate_devices(backend, _detect_visible_devices(backend))


def get_bench_lite_dir(current_file: Path) -> Path:
    """Locate the bench_lite directory from a runner file."""
    current = current_file.resolve()
    for parent in current.parents:
        candidate = parent / "benchmark" / "akg_kernels_bench_lite"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate akg_kernels_bench_lite from {current_file}")


def discover_bench_lite_cases(
    bench_lite_dir: Path,
    tiers: Optional[List[str]] = None,
    cases: Optional[List[str]] = None,
    filter_pattern: Optional[str] = None,
    skip_npu: bool = True,
) -> DiscoveryResult:
    """Discover benchmark_lite cases and capture skipped cases explicitly."""
    discovered: List[BenchLiteCase] = []
    skipped: List[BenchLiteCase] = []

    if not bench_lite_dir.exists():
        return DiscoveryResult(cases=[], skipped_cases=[])

    if tiers:
        selected_tiers = tiers
    else:
        # Auto-discover all t{N} directories, consistent with performance evaluation path
        selected_tiers = sorted(
            d.name for d in bench_lite_dir.iterdir()
            if d.is_dir() and d.name.startswith("t") and d.name[1:].isdigit()
        ) or list(DEFAULT_TIERS)
    selected_cases = set(cases) if cases else None
    keyword = filter_pattern.lower() if filter_pattern else None

    for tier in selected_tiers:
        tier_dir = bench_lite_dir / tier
        if not tier_dir.exists():
            continue

        for py_file in sorted(tier_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            case_name = py_file.stem
            if selected_cases is not None and case_name not in selected_cases:
                continue
            if keyword is not None and keyword not in case_name.lower():
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception as exc:
                skipped.append(
                    BenchLiteCase(
                        tier=tier,
                        case_name=case_name,
                        file_path=py_file,
                        requires_npu=False,
                        skip_reason=f"read_error: {exc}",
                    )
                )
                continue

            requires_npu = "torch_npu" in content
            if skip_npu and requires_npu:
                skipped.append(
                    BenchLiteCase(
                        tier=tier,
                        case_name=case_name,
                        file_path=py_file,
                        requires_npu=True,
                        skip_reason="requires torch_npu",
                    )
                )
                continue

            discovered.append(
                BenchLiteCase(
                    tier=tier,
                    case_name=case_name,
                    file_path=py_file,
                    requires_npu=requires_npu,
                    skip_reason=None,
                )
            )

    return DiscoveryResult(cases=discovered, skipped_cases=skipped)


def convert_case_source_for_backend(content: str, backend: str) -> str:
    """Convert generic case source for target backend when needed.

    Only replaces device literals and ``.cpu()`` calls that appear outside
    of comments (lines starting with ``#``) to reduce false positives.
    """
    _DEVICE_TARGETS = {"gpu": "cuda", "npu": "npu"}
    target = _DEVICE_TARGETS.get(backend)
    if not target:
        return content

    lines = content.split("\n")
    result_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            result_lines.append(line)
            continue
        line = re.sub(r"device='cpu'", f"device='{target}'", line)
        line = re.sub(r'device="cpu"', f'device="{target}"', line)
        # Match .cpu() when preceded by a word char or closing paren/bracket
        line = re.sub(r"(?<=[\w)\]])\.cpu\(\)", f".{target}()", line)
        result_lines.append(line)
    return "\n".join(result_lines)


def read_case_source(file_path: Path, backend: str) -> str:
    """Read a bench_lite case file using utf-8."""
    content = file_path.read_text(encoding="utf-8")
    return convert_case_source_for_backend(content, backend)


def build_task_specs(
    cases: List[BenchLiteCase],
    backend: str,
    backend_config: Dict[str, object],
    pass_n: int,
) -> List[Dict[str, object]]:
    """Build task specs for LangGraph tasks."""
    specs: List[Dict[str, object]] = []
    for case in cases:
        task_desc = read_case_source(case.file_path, backend)
        for attempt_id in range(1, pass_n + 1):
            task_id = f"{backend}_{case.tier}_{case.case_name}_attempt{attempt_id}"
            specs.append(
                {
                    "case": case,
                    "attempt_id": attempt_id,
                    "task_id": task_id,
                    "op_name": task_id,
                    "task_desc": task_desc,
                    "dsl": backend_config["dsl"],
                    "backend": backend_config["backend"],
                    "arch": backend_config["arch"],
                }
            )
    return specs


def _parse_result_op_name(op_name: str) -> Optional[Tuple[str, str, str, int]]:
    match = re.match(r"^(.+?)_(.+?)_(.+?)_attempt(\d+)$", op_name)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), int(match.group(4))


def _extract_failure_reason(item: object) -> Optional[str]:
    if isinstance(item, dict):
        for key in ("error", "reason", "message", "detail"):
            value = item.get(key)
            if value:
                return str(value)
    return None


def aggregate_correctness_results(
    raw_results: List[object],
    cases: List[BenchLiteCase],
    pass_n: int,
    backend: str,
) -> List[Dict[str, object]]:
    """Aggregate raw task results into correctness-oriented case results."""
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

    for item in raw_results:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        op_name, success = item[0], bool(item[1])
        parsed = _parse_result_op_name(str(op_name))
        if not parsed:
            continue
        _, tier, case_name, attempt_id = parsed
        grouped.setdefault((tier, case_name), []).append(
            {
                "attempt_id": attempt_id,
                "success": success,
                "task_id": str(op_name),
                "failure_reason": _extract_failure_reason(item[2]) if len(item) > 2 else None,
            }
        )

    aggregated: List[Dict[str, object]] = []
    for case in cases:
        attempts = sorted(
            grouped.get((case.tier, case.case_name), []),
            key=lambda value: int(value["attempt_id"]),
        )
        success_count = sum(1 for attempt in attempts if attempt["success"])
        failure_reason = None
        if success_count == 0 and attempts:
            failure_reason = next(
                (attempt["failure_reason"] for attempt in attempts if attempt.get("failure_reason")),
                "all attempts failed",
            )

        aggregated.append(
            {
                "tier": case.tier,
                "case_name": case.case_name,
                "pass_n": pass_n,
                "attempts": attempts,
                "success_count": success_count,
                "pass_rate": success_count / pass_n if pass_n > 0 else 0.0,
                "overall_success": success_count > 0,
                "failure_reason": failure_reason,
            }
        )

    return aggregated


def _serialize_skipped_case(case: BenchLiteCase, backend: Optional[str] = None) -> Dict[str, object]:
    payload = {
        "tier": case.tier,
        "case_name": case.case_name,
        "file_path": str(case.file_path),
        "requires_npu": case.requires_npu,
        "skip_reason": case.skip_reason,
    }
    if backend is not None:
        payload["backend"] = backend
    return payload


def compute_summary(
    results: List[Dict[str, object]],
    total_elapsed: float,
    skipped_cases: List[BenchLiteCase],
    environment_error: Optional[str] = None,
) -> Dict[str, object]:
    """Compute correctness summary for one backend."""
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result["overall_success"])
    failed_cases = total_cases - passed_cases
    total_attempts = sum(int(result["pass_n"]) for result in results)
    successful_attempts = sum(int(result["success_count"]) for result in results)

    tier_stats: Dict[str, Dict[str, object]] = {}
    for result in results:
        tier = str(result["tier"])
        stats = tier_stats.setdefault(
            tier,
            {
                "total_cases": 0,
                "passed_cases": 0,
                "total_attempts": 0,
                "successful_attempts": 0,
            },
        )
        stats["total_cases"] = int(stats["total_cases"]) + 1
        stats["passed_cases"] = int(stats["passed_cases"]) + (1 if result["overall_success"] else 0)
        stats["total_attempts"] = int(stats["total_attempts"]) + int(result["pass_n"])
        stats["successful_attempts"] = int(stats["successful_attempts"]) + int(result["success_count"])

    for stats in tier_stats.values():
        total = int(stats["total_cases"])
        attempts = int(stats["total_attempts"])
        stats["case_pass_rate"] = int(stats["passed_cases"]) / total if total else 0.0
        stats["attempt_pass_rate"] = int(stats["successful_attempts"]) / attempts if attempts else 0.0

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "case_pass_rate": passed_cases / total_cases if total_cases else 0.0,
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "attempt_pass_rate": successful_attempts / total_attempts if total_attempts else 0.0,
        "total_wall_time": total_elapsed,
        "tier_stats": tier_stats,
        "skipped_cases": [_serialize_skipped_case(case) for case in skipped_cases],
        "environment_error": environment_error,
    }


def _collect_system_info() -> Dict[str, Any]:
    """Collect hardware and software environment details."""
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }

    try:
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
    except Exception:
        info["torch_version"] = "unavailable"
        return info

    try:
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
            gpu_count = torch.cuda.device_count()
            info["gpu_count"] = gpu_count
            info["gpu_devices"] = []
            for i in range(gpu_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    info["gpu_devices"].append({
                        "index": i,
                        "name": props.name,
                        "total_memory_mb": round(props.total_memory / (1024 * 1024)),
                        "compute_capability": f"{props.major}.{props.minor}",
                    })
                except Exception:
                    info["gpu_devices"].append({"index": i, "error": "query failed"})
        else:
            info["cuda_version"] = None
            info["gpu_count"] = 0
    except Exception:
        pass

    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            info["npu_available"] = True
        else:
            info["npu_available"] = False
    except Exception:
        pass

    return info


def build_environment_info(backend: str, backend_config: Dict[str, object], devices: List[int]) -> Dict[str, object]:
    """Build environment metadata for JSON output."""
    env: Dict[str, object] = {
        "framework": "torch",
        "dsl": backend_config["dsl"],
        "backend": backend_config["backend"],
        "visible_devices": devices,
    }
    env["system"] = _collect_system_info()
    return env


def combine_backend_payloads(backend_payloads: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Combine per-backend payloads for --backend all."""
    combined_results: List[Dict[str, object]] = []
    combined_skips: List[Dict[str, object]] = []
    total_cases = 0
    passed_cases = 0
    failed_cases = 0
    total_attempts = 0
    successful_attempts = 0
    total_wall_time = 0.0
    tier_stats: Dict[str, Dict[str, object]] = {}
    environment_errors: Dict[str, str] = {}

    for backend, payload in backend_payloads.items():
        summary = payload.get("summary", {})
        total_cases += int(summary.get("total_cases", 0))
        passed_cases += int(summary.get("passed_cases", 0))
        failed_cases += int(summary.get("failed_cases", 0))
        total_attempts += int(summary.get("total_attempts", 0))
        successful_attempts += int(summary.get("successful_attempts", 0))
        total_wall_time += float(summary.get("total_wall_time", 0.0))
        if summary.get("environment_error"):
            environment_errors[backend] = str(summary["environment_error"])

        for skipped in summary.get("skipped_cases", []):
            entry = dict(skipped)
            entry["backend"] = backend
            combined_skips.append(entry)

        for result in payload.get("results", []):
            entry = dict(result)
            entry["backend"] = backend
            combined_results.append(entry)

        for tier, stats in summary.get("tier_stats", {}).items():
            combined = tier_stats.setdefault(
                tier,
                {
                    "total_cases": 0,
                    "passed_cases": 0,
                    "total_attempts": 0,
                    "successful_attempts": 0,
                },
            )
            combined["total_cases"] += int(stats.get("total_cases", 0))
            combined["passed_cases"] += int(stats.get("passed_cases", 0))
            combined["total_attempts"] += int(stats.get("total_attempts", 0))
            combined["successful_attempts"] += int(stats.get("successful_attempts", 0))

    for stats in tier_stats.values():
        total = int(stats["total_cases"])
        attempts = int(stats["total_attempts"])
        stats["case_pass_rate"] = int(stats["passed_cases"]) / total if total else 0.0
        stats["attempt_pass_rate"] = int(stats["successful_attempts"]) / attempts if attempts else 0.0

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "case_pass_rate": passed_cases / total_cases if total_cases else 0.0,
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "attempt_pass_rate": successful_attempts / total_attempts if total_attempts else 0.0,
        "total_wall_time": total_wall_time,
        "tier_stats": tier_stats,
        "skipped_cases": combined_skips,
        "environment_error": environment_errors if environment_errors else None,
        "results": combined_results,
    }


def print_backend_summary(backend_name: str, summary: Dict[str, object], results: List[Dict[str, object]]) -> None:
    """Print a human-readable correctness summary."""
    print("\n" + "=" * 80)
    print(f"Benchmark Lite Correctness Summary - {backend_name}")
    print("=" * 80)
    print(f"Total Cases: {summary.get('total_cases', 0)}")
    print(f"Passed Cases: {summary.get('passed_cases', 0)} ({summary.get('case_pass_rate', 0.0):.1%})")
    print(f"Failed Cases: {summary.get('failed_cases', 0)}")
    print(f"Total Attempts: {summary.get('total_attempts', 0)}")
    print(f"Successful Attempts: {summary.get('successful_attempts', 0)} ({summary.get('attempt_pass_rate', 0.0):.1%})")
    print(f"Total Wall Time: {summary.get('total_wall_time', 0.0):.2f}s")

    if summary.get("environment_error"):
        print(f"Environment Error: {summary['environment_error']}")

    skipped_cases = summary.get("skipped_cases", [])
    if skipped_cases:
        print(f"Skipped Cases: {len(skipped_cases)}")
        for skipped in skipped_cases:
            print(f"  - {skipped['tier']}/{skipped['case_name']}: {skipped['skip_reason']}")

    if summary.get("tier_stats"):
        print("\nPer-Tier Statistics:")
        print("-" * 80)
        for tier in sorted(summary["tier_stats"].keys()):
            stats = summary["tier_stats"][tier]
            print(
                f"{tier}: {stats.get('passed_cases', 0)}/{stats.get('total_cases', 0)} cases "
                f"({stats.get('case_pass_rate', 0.0):.1%}), "
                f"{stats.get('successful_attempts', 0)}/{stats.get('total_attempts', 0)} attempts "
                f"({stats.get('attempt_pass_rate', 0.0):.1%})"
            )

    if results:
        print("\nDetailed Results:")
        print("-" * 80)
        print(f"{'No.':<5} {'Tier':<6} {'Case':<30} {'Pass@N':<10} {'Result':<10}")
        print("-" * 80)
        for index, result in enumerate(results, start=1):
            case_result = "PASS" if result["overall_success"] else "FAIL"
            pass_n_str = f"{result['success_count']}/{result['pass_n']}"
            print(f"{index:<5} {result['tier']:<6} {result['case_name']:<30} {pass_n_str:<10} {case_result:<10}")
        print("-" * 80)


# ------------------------------------------------------------------------------
# Phase 2: Submission extraction from Agent results
# ------------------------------------------------------------------------------

DEFAULT_PERF_RTOL = 1e-2
DEFAULT_PERF_ATOL = 1e-2
DEFAULT_PERF_WARMUP = 10
DEFAULT_PERF_ITERATIONS = 100
DEFAULT_PERF_NUM_TRIALS = 3
DEFAULT_PERF_TIMEOUT = 300
CV_STABLE_THRESHOLD = 0.10

TIER_WEIGHTS: Dict[str, float] = {
    "t1": 1.0,
    "t2": 1.5,
    "t3": 2.0,
    "t4": 2.5,
    "t5": 3.0,
}


def _get_tier_weight(tier: str) -> float:
    return TIER_WEIGHTS.get(tier, 1.0)


def _compute_case_score(speedup: float) -> float:
    """Raw score (0~100) from speedup ratio."""
    if speedup <= 0:
        return 0.0
    if speedup < 1.0:
        return 60.0 * speedup
    bonus = min(speedup - 1.0, 4.0) / 4.0 * 40.0
    return 60.0 + bonus


def _compute_weighted_score(tier: str, speedup: float) -> float:
    return _compute_case_score(speedup) * _get_tier_weight(tier)


DEFAULT_TEAM_NAME = "aikg_agent"

# Regex for safe team names: alphanumeric, hyphens, underscores only
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_team_name(team_name: str) -> str:
    """Validate team_name to prevent path traversal.

    Raises:
        ValueError: If team_name contains path separators, parent refs, or
            characters outside ``[A-Za-z0-9_-]``.
    """
    if not team_name:
        raise ValueError("--team-name must not be empty")
    if not _SAFE_NAME_RE.match(team_name):
        raise ValueError(
            f"--team-name contains invalid characters: {team_name!r}. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )
    return team_name


def extract_submissions_from_results(
    raw_results: List[object],
    cases: List[BenchLiteCase],
    bench_lite_dir: Path,
    submission_dir: Path,
    backend: str,
    team_name: str = DEFAULT_TEAM_NAME,
) -> Path:
    """Extract generated ModelNew code from raw Agent results and write submission files.

    For each case with at least one successful attempt, pick the first successful
    attempt's ``coder_code`` and write it to ``{submission_dir}/{team_name}/{tier}/{case}.py``.

    Returns:
        The team directory (``{submission_dir}/{team_name}``).

    Raises:
        ValueError: If *team_name* fails path-safety validation.
    """
    validate_team_name(team_name)
    team_dir = submission_dir / team_name
    # Clean previous submissions to prevent stale files from polluting this run
    if team_dir.exists():
        shutil.rmtree(team_dir)
    case_lookup: Dict[Tuple[str, str], BenchLiteCase] = {
        (c.tier, c.case_name): c for c in cases
    }

    best_code: Dict[Tuple[str, str], str] = {}

    for item in raw_results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        op_name, success, state = item[0], bool(item[1]), item[2]
        if not success:
            continue
        parsed = _parse_result_op_name(str(op_name))
        if not parsed:
            continue
        _, tier, case_name, _attempt_id = parsed
        key = (tier, case_name)
        if key in best_code:
            continue
        code = None
        if isinstance(state, dict):
            code = state.get("coder_code")
        if code and key in case_lookup:
            best_code[key] = code

    for (tier, case_name), code in best_code.items():
        tier_dir = team_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        (tier_dir / f"{case_name}.py").write_text(code, encoding="utf-8")

    meta = {
        "team_name": team_name,
        "backend": backend,
        "timestamp": datetime.now().isoformat(),
        "cases_extracted": len(best_code),
    }
    team_dir.mkdir(parents=True, exist_ok=True)
    (team_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return team_dir


# ------------------------------------------------------------------------------
# Phase 3: Performance evaluation (reuses logic from tools/run_bench.py)
# ------------------------------------------------------------------------------

def _load_module_from_path(py_path: Path, module_name: str) -> Any:
    """Load a Python module from file path without polluting ``sys.modules``."""
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    # Temporarily register so relative imports inside the module work,
    # then remove to avoid polluting the global module namespace.
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    sys.modules.pop(module_name, None)
    return mod


def _get_torch_device(backend: Optional[str] = None) -> str:
    """Return the torch device string for the given backend.

    When *backend* is provided, returns the corresponding device string
    directly (``gpu`` -> ``cuda``, ``npu`` -> ``npu``, ``cpu`` -> ``cpu``).
    When *backend* is ``None``, auto-detects the best available device
    (legacy behaviour, kept for backward compatibility).
    """
    _BACKEND_TO_DEVICE = {"gpu": "cuda", "npu": "npu", "cpu": "cpu"}
    if backend is not None:
        return _BACKEND_TO_DEVICE.get(backend, backend)

    # Auto-detect (legacy)
    try:
        import torch  # type: ignore
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _sync_torch_device(device: str) -> None:
    try:
        import torch  # type: ignore
        if device == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def _check_correctness(
    ref_outputs: Any,
    sol_outputs: Any,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """Verify solution outputs against reference outputs."""
    import torch  # type: ignore

    if isinstance(ref_outputs, torch.Tensor):
        ref_list = [ref_outputs]
        sol_list = [sol_outputs] if isinstance(sol_outputs, torch.Tensor) else [sol_outputs]
    elif isinstance(ref_outputs, (tuple, list)):
        ref_list = list(ref_outputs)
        sol_list = list(sol_outputs) if isinstance(sol_outputs, (tuple, list)) else [sol_outputs]
    else:
        return {"correct": False, "max_abs_diff": float("inf"), "max_rel_diff": float("inf"),
                "detail": f"Unsupported output type: {type(ref_outputs)}"}

    if len(ref_list) != len(sol_list):
        return {"correct": False, "max_abs_diff": float("inf"), "max_rel_diff": float("inf"),
                "detail": f"Output count mismatch: ref={len(ref_list)}, sol={len(sol_list)}"}

    max_abs = 0.0
    max_rel = 0.0
    for i, (ref_t, sol_t) in enumerate(zip(ref_list, sol_list)):
        if not isinstance(ref_t, torch.Tensor) or not isinstance(sol_t, torch.Tensor):
            return {"correct": False, "max_abs_diff": float("inf"), "max_rel_diff": float("inf"),
                    "detail": f"Output[{i}] is not a Tensor"}
        if ref_t.shape != sol_t.shape:
            return {"correct": False, "max_abs_diff": float("inf"), "max_rel_diff": float("inf"),
                    "detail": f"Output[{i}] shape mismatch: ref={ref_t.shape}, sol={sol_t.shape}"}
        ref_f = ref_t.float()
        sol_f = sol_t.float()
        if torch.isnan(sol_f).any() or torch.isinf(sol_f).any():
            return {"correct": False, "max_abs_diff": float("inf"), "max_rel_diff": float("inf"),
                    "detail": f"Output[{i}] contains NaN or Inf"}
        abs_diff = (ref_f - sol_f).abs()
        rel_diff = abs_diff / (ref_f.abs() + 1e-8)
        max_abs = max(max_abs, abs_diff.max().item())
        max_rel = max(max_rel, rel_diff.max().item())
        # Strict AND semantics: absolute AND relative tolerance must both pass
        if max_abs > atol or max_rel > rtol:
            detail = f"max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}"
            return {"correct": False, "max_abs_diff": max_abs, "max_rel_diff": max_rel, "detail": detail}

    return {"correct": True, "max_abs_diff": max_abs, "max_rel_diff": max_rel, "detail": "PASS"}


def _use_cuda_events(device: str) -> bool:
    """Return True when CUDA event-based timing is available and appropriate."""
    if device != "cuda":
        return False
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


def _measure_latency(
    fn: Any,
    args: list,
    warmup_runs: int = DEFAULT_PERF_WARMUP,
    iterations: int = DEFAULT_PERF_ITERATIONS,
    num_trials: int = DEFAULT_PERF_NUM_TRIALS,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure kernel latency with warmup, multi-trial median.

    Uses CUDA events for GPU-side timing when available (avoids CPU-GPU
    synchronization overhead).  Falls back to ``time.perf_counter`` on
    CPU / NPU backends.
    """
    device = _get_torch_device(backend)
    use_events = _use_cuda_events(device)

    for _ in range(warmup_runs):
        fn(*args)
    _sync_torch_device(device)

    trial_times: List[float] = []

    if use_events:
        import torch  # type: ignore
        for _ in range(num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iterations):
                fn(*args)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            trial_times.append(elapsed_ms / iterations)
    else:
        for _ in range(num_trials):
            _sync_torch_device(device)
            start = time.perf_counter()
            for _ in range(iterations):
                fn(*args)
            _sync_torch_device(device)
            elapsed = time.perf_counter() - start
            trial_times.append((elapsed / iterations) * 1000)

    trial_times.sort()
    n = len(trial_times)
    if n % 2 == 1:
        median_ms = trial_times[n // 2]
    else:
        median_ms = (trial_times[n // 2 - 1] + trial_times[n // 2]) / 2
    mean_ms = sum(trial_times) / n
    std_ms = math.sqrt(sum((t - mean_ms) ** 2 for t in trial_times) / len(trial_times))
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0

    return {
        "median_ms": median_ms,
        "mean_ms": round(mean_ms, 6),
        "min_ms": trial_times[0],
        "max_ms": trial_times[-1],
        "std_ms": round(std_ms, 6),
        "cv": round(cv, 6),
        "stable": cv <= CV_STABLE_THRESHOLD,
        "timing_method": "cuda_events" if use_events else "cpu_perf_counter",
        "all_trials_ms": [round(t, 6) for t in trial_times],
    }


def _clear_gpu_cache() -> None:
    """Best-effort GPU cache cleanup between cases."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
    except Exception:
        pass


def _eval_single_case_inner(
    ref_path: Path,
    sol_path: Path,
    tier: str,
    rtol: float = DEFAULT_PERF_RTOL,
    atol: float = DEFAULT_PERF_ATOL,
    warmup_runs: int = DEFAULT_PERF_WARMUP,
    iterations: int = DEFAULT_PERF_ITERATIONS,
    num_trials: int = DEFAULT_PERF_NUM_TRIALS,
    _phase_tracker: Optional[List[str]] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Core evaluation logic for a single case (no timeout wrapper).

    Args:
        _phase_tracker: Optional mutable list used by ``_eval_single_case``
            to track which phase the evaluation reached when a timeout occurs.
        backend: The target backend (``gpu``, ``npu``, ``cpu``).  When
            provided, the device is determined from the backend rather than
            auto-detected.
    """
    import torch  # type: ignore

    def _track(phase: str) -> None:
        if _phase_tracker is not None:
            _phase_tracker.append(phase)

    case_name = f"{tier}/{sol_path.stem}"
    result: Dict[str, Any] = {"case": case_name, "tier": tier, "case_name": sol_path.stem}

    try:
        _clear_gpu_cache()

        _track("loading_modules")
        ref_mod = _load_module_from_path(ref_path, f"ref_{ref_path.stem}")
        sol_mod = _load_module_from_path(sol_path, f"sol_{sol_path.stem}")

        RefModel = ref_mod.Model
        SolModel = sol_mod.ModelNew
        get_inputs = ref_mod.get_inputs
        get_init_inputs = ref_mod.get_init_inputs

        _track("model_init")
        device = _get_torch_device(backend)
        init_args = get_init_inputs()

        ref_model = RefModel(*init_args)
        sol_model = SolModel(*init_args)

        if device != "cpu":
            ref_model = ref_model.to(device)
            sol_model = sol_model.to(device)

        ref_model.eval()
        sol_model.eval()

        inputs = get_inputs()
        if device != "cpu":
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

        _track("correctness_check")
        with torch.no_grad():
            ref_out = ref_model(*inputs)
            sol_out = sol_model(*inputs)

        corr = _check_correctness(ref_out, sol_out, rtol=rtol, atol=atol)
        result["correctness"] = corr["correct"]
        result["max_abs_diff"] = corr["max_abs_diff"]
        result["max_rel_diff"] = corr["max_rel_diff"]
        result["correctness_detail"] = corr["detail"]

        if not corr["correct"]:
            result["status"] = "fail"
            result["score"] = 0.0
            result["weighted_score"] = 0.0
            result["error"] = corr["detail"]
            return result

        def ref_fn(*a):
            return ref_model(*a)

        def sol_fn(*a):
            return sol_model(*a)

        _track("measuring_baseline")
        with torch.no_grad():
            baseline = _measure_latency(ref_fn, inputs, warmup_runs, iterations, num_trials, backend)
            _track("measuring_solution")
            solution = _measure_latency(sol_fn, inputs, warmup_runs, iterations, num_trials, backend)

        speedup = baseline["median_ms"] / solution["median_ms"] if solution["median_ms"] > 0 else 0.0
        measurement_stable = baseline["stable"] and solution["stable"]

        result["status"] = "pass"
        result["baseline_ms"] = round(baseline["median_ms"], 4)
        result["solution_ms"] = round(solution["median_ms"], 4)
        result["speedup"] = round(speedup, 4)
        result["score"] = round(_compute_case_score(speedup), 2)
        result["weighted_score"] = round(_compute_weighted_score(tier, speedup), 2)
        result["measurement_stable"] = measurement_stable
        result["timing_method"] = solution["timing_method"]
        result["baseline_detail"] = baseline
        result["solution_detail"] = solution
        result["error"] = None

    except Exception as exc:
        result["status"] = "error"
        result["correctness"] = False
        result["score"] = 0.0
        result["weighted_score"] = 0.0
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    return result


def _eval_worker_target(
    ref_path: Path, sol_path: Path, tier: str,
    rtol: float, atol: float, warmup_runs: int, iterations: int,
    num_trials: int, backend: Optional[str],
    result_queue: multiprocessing.Queue,
    phase_value: multiprocessing.Value,
) -> None:
    """Target function for the evaluation subprocess."""
    _PHASE_IDS = {
        "loading_modules": 1, "model_init": 2, "correctness_check": 3,
        "measuring_baseline": 4, "measuring_solution": 5,
    }

    class SharedPhaseTracker(list):
        """A list that also writes phase IDs to shared memory."""
        def append(self, phase: str) -> None:
            super().append(phase)
            phase_value.value = _PHASE_IDS.get(phase, 0)

    try:
        result = _eval_single_case_inner(
            ref_path, sol_path, tier,
            rtol, atol, warmup_runs, iterations, num_trials,
            SharedPhaseTracker(), backend,
        )
        result_queue.put(result)
    except Exception as exc:
        result_queue.put({
            "case": f"{tier}/{sol_path.stem}",
            "tier": tier,
            "case_name": sol_path.stem,
            "status": "error",
            "correctness": False,
            "score": 0.0,
            "weighted_score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        })


_PHASE_NAMES = {
    0: "startup", 1: "loading_modules", 2: "model_init",
    3: "correctness_check", 4: "measuring_baseline", 5: "measuring_solution",
}


def _eval_single_case(
    ref_path: Path,
    sol_path: Path,
    tier: str,
    rtol: float = DEFAULT_PERF_RTOL,
    atol: float = DEFAULT_PERF_ATOL,
    warmup_runs: int = DEFAULT_PERF_WARMUP,
    iterations: int = DEFAULT_PERF_ITERATIONS,
    num_trials: int = DEFAULT_PERF_NUM_TRIALS,
    timeout: int = DEFAULT_PERF_TIMEOUT,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single case with timeout protection via subprocess isolation.

    Runs ``_eval_single_case_inner`` in a separate process so that:
    - On timeout the worker process is terminated, freeing GPU/NPU resources.
    - A stuck evaluation cannot pollute timing or resource state of subsequent cases.
    """
    case_name = f"{tier}/{sol_path.stem}"

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    phase_value = ctx.Value("i", 0)

    proc = ctx.Process(
        target=_eval_worker_target,
        args=(
            ref_path, sol_path, tier,
            rtol, atol, warmup_runs, iterations, num_trials,
            backend, result_queue, phase_value,
        ),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        last_phase = _PHASE_NAMES.get(phase_value.value, "unknown")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        return {
            "case": case_name,
            "tier": tier,
            "case_name": sol_path.stem,
            "status": "error",
            "correctness": False,
            "score": 0.0,
            "weighted_score": 0.0,
            "error": f"Timeout: exceeded {timeout}s limit (last phase: {last_phase})",
        }

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        return {
            "case": case_name,
            "tier": tier,
            "case_name": sol_path.stem,
            "status": "error",
            "correctness": False,
            "score": 0.0,
            "weighted_score": 0.0,
            "error": f"Worker process exited with code {proc.exitcode} without producing a result",
        }
    return result


def run_performance_evaluation(
    team_dir: Path,
    bench_lite_dir: Path,
    warmup_runs: int = DEFAULT_PERF_WARMUP,
    iterations: int = DEFAULT_PERF_ITERATIONS,
    num_trials: int = DEFAULT_PERF_NUM_TRIALS,
    rtol: float = DEFAULT_PERF_RTOL,
    atol: float = DEFAULT_PERF_ATOL,
    timeout: int = DEFAULT_PERF_TIMEOUT,
    backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run performance evaluation on extracted submissions.

    Iterates over ``team_dir/{tier}/*.py``, pairs each file with the
    corresponding reference in ``bench_lite_dir/{tier}/{name}.py``, and
    runs correctness + performance measurement with per-case timeout.

    Args:
        backend: The target backend (``gpu``, ``npu``, ``cpu``).  Passed
            through to ``_eval_single_case`` to ensure the correct device
            is used for performance measurement.

    Returns:
        List of per-case result dicts.
    """
    perf_results: List[Dict[str, Any]] = []

    tiers = sorted(
        d.name for d in bench_lite_dir.iterdir()
        if d.is_dir() and d.name.startswith("t") and d.name[1:].isdigit()
    )

    for tier in tiers:
        sol_tier_dir = team_dir / tier
        ref_tier_dir = bench_lite_dir / tier

        if not sol_tier_dir.exists():
            continue

        for sol_file in sorted(sol_tier_dir.glob("*.py")):
            if sol_file.name == "__init__.py":
                continue

            ref_file = ref_tier_dir / sol_file.name
            if not ref_file.exists():
                perf_results.append({
                    "case": f"{tier}/{sol_file.stem}",
                    "tier": tier,
                    "case_name": sol_file.stem,
                    "status": "error",
                    "correctness": False,
                    "score": 0.0,
                    "weighted_score": 0.0,
                    "error": f"Reference not found: {sol_file.name}",
                })
                continue

            print(f"  [PERF] {tier}/{sol_file.stem} ...", end=" ", flush=True)
            result = _eval_single_case(
                ref_path=ref_file,
                sol_path=sol_file,
                tier=tier,
                rtol=rtol,
                atol=atol,
                warmup_runs=warmup_runs,
                iterations=iterations,
                num_trials=num_trials,
                timeout=timeout,
                backend=backend,
            )
            icon = "PASS" if result["status"] == "pass" else "FAIL"
            extra = ""
            if result["status"] == "pass":
                stability = "" if result.get("measurement_stable", True) else " [UNSTABLE]"
                extra = f" speedup={result['speedup']}x score={result['weighted_score']}{stability}"
            elif result.get("error"):
                extra = f" {str(result['error'])[:80]}"
            print(f"{icon}{extra}")
            perf_results.append(result)

    return perf_results


# ------------------------------------------------------------------------------
# Phase 4: Full report and leaderboard
# ------------------------------------------------------------------------------

def compute_performance_summary(perf_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate performance summary from per-case perf results."""
    passed = [r for r in perf_results if r.get("status") == "pass"]
    total_weighted = sum(r.get("weighted_score", 0.0) for r in perf_results)
    avg_speedup = (
        sum(r["speedup"] for r in passed) / len(passed)
        if passed else 0.0
    )

    stable_count = sum(1 for r in passed if r.get("measurement_stable", True))
    unstable_count = len(passed) - stable_count

    tier_perf: Dict[str, Dict[str, Any]] = {}
    for r in perf_results:
        tier = r.get("tier", "unknown")
        stats = tier_perf.setdefault(tier, {
            "total": 0, "passed": 0, "total_weighted_score": 0.0, "speedups": [],
        })
        stats["total"] += 1
        if r.get("status") == "pass":
            stats["passed"] += 1
            stats["speedups"].append(r["speedup"])
        stats["total_weighted_score"] += r.get("weighted_score", 0.0)

    for stats in tier_perf.values():
        stats["total_weighted_score"] = round(stats["total_weighted_score"], 2)
        stats["avg_speedup"] = round(
            sum(stats["speedups"]) / len(stats["speedups"]) if stats["speedups"] else 0.0, 4
        )
        del stats["speedups"]

    return {
        "total_cases": len(perf_results),
        "passed_cases": len(passed),
        "failed_cases": len(perf_results) - len(passed),
        "total_weighted_score": round(total_weighted, 2),
        "avg_speedup": round(avg_speedup, 4),
        "stable_measurements": stable_count,
        "unstable_measurements": unstable_count,
        "tier_stats": tier_perf,
    }


def print_performance_summary(
    backend_name: str,
    perf_results: List[Dict[str, Any]],
    perf_summary: Dict[str, Any],
) -> None:
    """Print human-readable performance evaluation results."""
    print("\n" + "=" * 80)
    print(f"Benchmark Lite Performance Evaluation - {backend_name}")
    print("=" * 80)

    if not perf_results:
        print("No cases evaluated (no successful Agent outputs).")
        return

    print(f"\n{'No.':<5} {'Tier':<6} {'Case':<22} {'Status':<8} {'Speedup':<10} "
          f"{'Raw':<8} {'Weighted':<10} {'Base(ms)':<12} {'Sol(ms)':<12} {'Stable':<8}")
    print("-" * 110)

    for i, r in enumerate(perf_results, 1):
        status = r.get("status", "error")
        speedup_str = f"{r['speedup']:.2f}x" if status == "pass" else "-"
        raw_str = f"{r.get('score', 0.0):.1f}" if status == "pass" else "0.0"
        weighted_str = f"{r.get('weighted_score', 0.0):.1f}"
        baseline_str = f"{r['baseline_ms']:.4f}" if status == "pass" else "-"
        solution_str = f"{r['solution_ms']:.4f}" if status == "pass" else "-"
        stable_str = ("Y" if r.get("measurement_stable", True) else "N") if status == "pass" else "-"
        case_name = r.get("case_name", r.get("case", "?"))
        print(f"{i:<5} {r.get('tier', '?'):<6} {case_name:<22} {status.upper():<8} "
              f"{speedup_str:<10} {raw_str:<8} {weighted_str:<10} {baseline_str:<12} {solution_str:<12} {stable_str:<8}")

    print("-" * 110)
    print(f"\nTotal Weighted Score: {perf_summary.get('total_weighted_score', 0.0)}")
    print(f"Average Speedup: {perf_summary.get('avg_speedup', 0.0):.2f}x")
    print(f"Cases Evaluated: {perf_summary.get('passed_cases', 0)}/{perf_summary.get('total_cases', 0)}")
    if perf_summary.get("unstable_measurements", 0) > 0:
        print(f"Measurement Stability: {perf_summary['stable_measurements']} stable, "
              f"{perf_summary['unstable_measurements']} unstable (CV > {CV_STABLE_THRESHOLD:.0%})")

    if perf_summary.get("tier_stats"):
        print("\nPer-Tier Performance:")
        print("-" * 60)
        for tier in sorted(perf_summary["tier_stats"].keys()):
            ts = perf_summary["tier_stats"][tier]
            print(f"  {tier}: {ts['passed']}/{ts['total']} passed, "
                  f"avg speedup={ts['avg_speedup']:.2f}x, "
                  f"weighted score={ts['total_weighted_score']}")

    print("=" * 80)


def print_full_summary(
    backend_name: str,
    summary: Dict[str, object],
    results: List[Dict[str, object]],
    perf_results: Optional[List[Dict[str, Any]]],
    perf_summary: Optional[Dict[str, Any]],
    mode: str = "full",
) -> None:
    """Print combined correctness + performance summary."""
    mode_label = {"performance": "PERFORMANCE", "full": "FULL"}.get(mode, mode.upper())
    print("\n" + "=" * 80)
    print(f"Benchmark Lite {mode_label} Evaluation - {backend_name}")
    print("=" * 80)

    print(f"\nPhase 1: Correctness (Pass@N)")
    print(f"  Total Cases: {summary.get('total_cases', 0)}    "
          f"Passed: {summary.get('passed_cases', 0)} ({summary.get('case_pass_rate', 0.0):.1%})    "
          f"Failed: {summary.get('failed_cases', 0)}")

    if perf_results and perf_summary:
        print(f"\nPhase 2: Performance (Speedup + Score)")
        print(f"  {'No.':<5} {'Tier':<6} {'Case':<25} {'Speedup':<10} {'Raw Score':<12} {'Weighted':<10}")
        print(f"  {'-' * 68}")
        for i, r in enumerate(perf_results, 1):
            if r.get("status") == "pass":
                print(f"  {i:<5} {r.get('tier', '?'):<6} {r.get('case_name', '?'):<25} "
                      f"{r.get('speedup', 0.0):.2f}x{'':<5} "
                      f"{r.get('score', 0.0):<12.1f} {r.get('weighted_score', 0.0):<10.1f}")
            else:
                reason = r.get("error", "FAIL")
                if reason and len(reason) > 25:
                    reason = reason[:22] + "..."
                print(f"  {i:<5} {r.get('tier', '?'):<6} {r.get('case_name', '?'):<25} "
                      f"{'FAIL':<10} {'':<12} {reason}")

        print(f"\nPhase 3: Summary")
        print(f"  Total Weighted Score: {perf_summary.get('total_weighted_score', 0.0)}")
        print(f"  Avg Speedup: {perf_summary.get('avg_speedup', 0.0):.2f}x")
        print(f"  Cases Evaluated: {perf_summary.get('passed_cases', 0)}/{perf_summary.get('total_cases', 0)} "
              f"(only correct cases scored)")
    else:
        print(f"\n  (Performance evaluation skipped)")

    print("=" * 80)


def build_full_output_payload(
    backend: str,
    mode: str,
    config_payload: Dict[str, object],
    environment: Dict[str, object],
    summary: Dict[str, object],
    results: List[Dict[str, object]],
    perf_results: Optional[List[Dict[str, Any]]] = None,
    perf_summary: Optional[Dict[str, Any]] = None,
    perf_config: Optional[Dict[str, Any]] = None,
    submission_dir: Optional[str] = None,
    backend_results: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build the complete JSON artifact for any mode."""
    payload: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "runner_version": RUNNER_VERSION,
        "mode": mode,
        "config": config_payload,
        "environment": environment,
        "summary": summary,
        "results": results,
    }

    if perf_results is not None:
        payload["performance_config"] = perf_config or {}
        payload["performance_results"] = perf_results
        payload["performance_summary"] = perf_summary or {}
        if submission_dir:
            payload["submission_dir"] = submission_dir

    if backend_results is not None:
        payload["backend_results"] = backend_results

    return payload


def write_leaderboard(
    perf_results: List[Dict[str, Any]],
    perf_summary: Dict[str, Any],
    output_path: Path,
    backend: str,
    team_name: str = DEFAULT_TEAM_NAME,
) -> None:
    """Write a leaderboard JSON file from evaluation results."""
    # Sort for deterministic output: by tier then case name
    sorted_results = sorted(perf_results, key=lambda r: (r.get("tier", ""), r.get("case_name", "")))

    cases_by_tier: Dict[str, List[Dict[str, Any]]] = {}
    for r in sorted_results:
        tier = r.get("tier", "unknown")
        entry: Dict[str, Any] = {
            "case": r.get("case", "?"),
            "status": r.get("status", "error"),
            "speedup": r.get("speedup", 0.0),
            "weighted_score": r.get("weighted_score", 0.0),
        }
        if r.get("backend"):
            entry["backend"] = r["backend"]
        if r.get("error"):
            entry["error"] = r["error"]
        cases_by_tier.setdefault(tier, []).append(entry)

    leaderboard = {
        "generated_at": datetime.now().isoformat(),
        "total_teams": 1,
        "ranking": [{
            "rank": 1,
            "team_name": team_name,
            "total_weighted_score": perf_summary.get("total_weighted_score", 0.0),
            "passed": perf_summary.get("passed_cases", 0),
            "total": perf_summary.get("total_cases", 0),
            "avg_speedup": perf_summary.get("avg_speedup", 0.0),
            "backend": backend,
            "cases_by_tier": dict(sorted(cases_by_tier.items())),
        }],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)
    print(f"Leaderboard saved to: {output_path}")
