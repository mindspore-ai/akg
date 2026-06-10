# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""Single-source-of-truth config loader for ``akg_cli worker``.

Replaces four sibling helpers (``load_default_port`` / ``load_default_dsl``
/ ``load_default_backend`` / ``_worker_setting``) that each re-resolved
and re-read the same yaml. One ``WorkerConfig.load(path)`` call returns a
frozen dataclass; everything downstream reads from it instead of poking
yaml again."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from akg_agents.op.utils.arch_normalize import (
    normalize_ascend_arch_name,
    normalize_cpu_arch_name,
    normalize_cuda_arch_name,
)


@dataclass(frozen=True)
class WorkerTiming:
    """``worker.*`` timing knobs from config.yaml. All in seconds (float)."""
    ready_timeout: float = 60.0          # 总等 daemon /status ready 多久
    ready_poll_interval: float = 5.0     # 心跳 tick 间隔
    ready_probe_timeout: float = 3.0     # 每次 /status probe 单次 timeout
    status_timeout: float = 3.0          # idle --status 探活的单次 timeout


@dataclass(frozen=True)
class WorkerConfig:
    """Top-level worker config — read once, passed around as a value object.

    All non-DSL fields have concrete defaults baked in（``port=9001``、
    ``backend="cuda"`` 等），所以 ``WorkerConfig`` 就是 single source of
    truth：callers 直接读 ``cfg.port`` / ``cfg.backend`` 不再各处 ``or
    "cuda"`` / ``else 9001`` 兜底。覆盖只允许向上 (CLI > env > yaml)，
    fallback 永远只这一处。

    ``dsl`` 保留 Optional —— ``None`` 有语义（"未指定 DSL"，触发 classify
    的 warn 而不是 fatal），不能用 "" 顶替。"""
    port: int = 9001
    backend: str = "cuda"
    arch: str = "a100"
    devices: str = "0"
    dsl: Optional[str] = None
    hosts: Dict[str, dict] = field(default_factory=dict)
    timing: WorkerTiming = field(default_factory=WorkerTiming)
    source_path: Optional[str] = None    # 解析 yaml 的绝对路径，diag 用

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "WorkerConfig":
        """Load from explicit path or default ``cwd/config.yaml``. yaml 缺
        失或字段缺失 → 用 dataclass 默认。绝不返回 None，callers 不用
        null-check。"""
        resolved = _resolve(config_path)
        if resolved is None:
            return cls()
        data = _load_yaml(resolved)
        if data is None:
            return cls(source_path=resolved)

        worker = data.get("worker") or {}
        defaults = data.get("defaults") or {}
        hosts = ((data.get("remote_worker") or {}).get("hosts") or {})

        port_v = _int_in_range(worker.get("port"), 1, 65535, cls.port)
        backend_v = _str_or(defaults.get("backend"), cls.backend).lower()
        arch_v = _str_or(defaults.get("arch"), cls.arch)
        devices_v = _str_or(defaults.get("devices"), cls.devices)
        dsl_raw = defaults.get("dsl")
        dsl_v: Optional[str] = str(dsl_raw) if isinstance(dsl_raw, str) else None

        # Timing defaults pulled FROM WorkerTiming so the two
        # dataclasses don't duplicate numbers.
        td = WorkerTiming()
        timing = WorkerTiming(
            ready_timeout=_float(worker.get("ready_timeout"), td.ready_timeout),
            ready_poll_interval=_float(worker.get("ready_poll_interval"),
                                       td.ready_poll_interval),
            ready_probe_timeout=_float(worker.get("ready_probe_timeout"),
                                       td.ready_probe_timeout),
            status_timeout=_float(worker.get("status_timeout"), td.status_timeout),
        )
        return cls(
            port=port_v, backend=backend_v, arch=arch_v, devices=devices_v,
            dsl=dsl_v, hosts=dict(hosts), timing=timing, source_path=resolved,
        )

    def host(self, alias: str) -> Optional[dict]:
        """Look up ``remote_worker.hosts.<alias>``. None if absent."""
        return self.hosts.get(alias)


def _resolve(config_path: Optional[str]) -> Optional[str]:
    if config_path is None:
        default = Path.cwd() / "config.yaml"
        return str(default) if default.is_file() else None
    return config_path if Path(config_path).is_file() else None


def _load_yaml(config_path: str) -> Optional[dict]:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[akg_cli] failed to read {config_path}: {e}", file=sys.stderr)
        return None


def _float(val, default: float) -> float:
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    return default


def _int_in_range(val, lo: int, hi: int, default: int) -> int:
    if isinstance(val, int) and lo <= val <= hi:
        return val
    return default


def _str_or(val, default: str) -> str:
    return str(val).strip() if isinstance(val, str) and str(val).strip() else default


def probe_local_arch(backend: str, device_id: int = 0) -> Optional[str]:
    """Best-effort local arch probe so ``akg_cli worker --start`` (no
    --remote-host) doesn't fall back to the baked ``a100`` default on
    hosts the operator forgot to flag.

    Per-backend dispatch:
      - ``ascend`` → ``npu-smi info`` main table → ``ascend<chip>``
      - ``cuda``   → ``nvidia-smi --query-gpu=name`` → normalized SKU
      - ``cpu``    → ``platform.machine()`` → x86_64 / aarch64

    Returns None on any failure (binary not on PATH, non-zero exit,
    unparseable output, unknown backend); caller falls through to
    ``cfg.arch``. Mirrors ``workspace_autoresearch/scripts/utils/
    hw_detect.derive_arch`` — kept duplicated rather than imported
    because the CLI package shouldn't reach into the workspace tree."""
    b = (backend or "").lower()
    if b == "ascend":
        return _probe_arch_ascend(device_id)
    if b == "cuda":
        return _probe_arch_cuda(device_id)
    if b == "cpu":
        return _probe_arch_cpu()
    return None


def _probe_arch_ascend(device_id: int) -> Optional[str]:
    try:
        r = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    m = re.search(rf"^\|\s*{int(device_id)}\s+(\S+)\s*\|", r.stdout, re.MULTILINE)
    if not m:
        return None
    return normalize_ascend_arch_name(m.group(1))


def _probe_arch_cuda(device_id: int) -> Optional[str]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             "-i", str(int(device_id))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    name = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    if not name:
        return None
    return normalize_cuda_arch_name(name)


def _probe_arch_cpu() -> Optional[str]:
    return normalize_cpu_arch_name()
