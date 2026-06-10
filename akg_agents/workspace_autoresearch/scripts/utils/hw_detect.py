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

"""Hardware detection: derive an AKG arch token from a local device id."""

from __future__ import annotations

import re
import subprocess
from typing import Optional

from akg_agents.op.utils.arch_normalize import (
    normalize_ascend_arch_name,
    normalize_cpu_arch_name,
    normalize_cuda_arch_name,
)


def derive_arch(device_id: int, backend: str = "ascend") -> Optional[str]:
    """Return the arch token for ``device_id`` on ``backend``.

    Returns None if the probe is unavailable or unparseable; callers decide
    whether that is fatal for their workflow.
    """
    backend = (backend or "ascend").lower()
    if backend == "ascend":
        return _derive_arch_ascend(device_id)
    if backend == "cuda":
        return _derive_arch_cuda(device_id)
    if backend == "cpu":
        return normalize_cpu_arch_name()
    return None


def probe_hint(backend: str) -> str:
    """User-facing probe-tool hint for arch derivation errors."""
    backend = (backend or "ascend").lower()
    return {
        "ascend": "is npu-smi on PATH?",
        "cuda": "is nvidia-smi on PATH?",
        "cpu": "platform.machine() returned empty",
    }.get(backend, f"no arch probe registered for backend={backend!r}")


def _derive_arch_ascend(device_id: int) -> Optional[str]:
    try:
        r = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    match = re.search(rf"^\|\s*{int(device_id)}\s+(\S+)\s*\|", r.stdout, re.MULTILINE)
    return normalize_ascend_arch_name(match.group(1)) if match else None


def _derive_arch_cuda(device_id: int) -> Optional[str]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             "-i", str(int(device_id))],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    name = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    return normalize_cuda_arch_name(name) if name else None
