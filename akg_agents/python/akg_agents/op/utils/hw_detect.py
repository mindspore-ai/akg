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

"""Hardware detection: derive an AKG arch token from a local device id.

Single source of truth shared by the workspace scaffold/verify scripts (via a
thin ``scripts/utils/hw_detect`` re-export shim) and the CLI worker, so the
npu-smi / nvidia-smi probing lives in exactly one place.
"""

from __future__ import annotations

import re
import subprocess
from typing import Optional

from .arch_normalize import (
    ascend_soc_version,
    normalize_ascend_arch_name,
    normalize_cpu_arch_name,
    normalize_cuda_arch_name,
)


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
    # npu-smi device tables differ across SoCs: 910B/310P pack "<id> <ChipName>"
    # in one cell ("| 0     910B3 |"); A5/950 split them with a column separator
    # ("| 0 | Ascend950PR |"). The optional ``\|?`` captures the chip name in
    # both. (Process/sub rows sit below the device table, so the first match for
    # an id is always its device row.)
    match = re.search(rf"^\|\s*{int(device_id)}\s*\|?\s*(\S+)", r.stdout,
                      re.MULTILINE)
    if not match:
        return None
    chip_name = match.group(1)
    arch = normalize_ascend_arch_name(chip_name)
    # Some SoCs only become a concrete SOC_VERSION once the board "NPU Name"
    # variant is appended — e.g. Ascend950PR -> ascend950pr (no SOC_VERSION)
    # vs ascend950pr_957b. Probe the variant ONLY when the bare family name
    # doesn't resolve, so 910B/310P stay a single npu-smi call (no regression).
    if arch and ascend_soc_version(arch) is None:
        variant = _query_board_variant(int(device_id))
        if variant and variant.lower() not in chip_name.lower():
            arch = normalize_ascend_arch_name(f"{chip_name}_{variant}")
    return arch


def _query_board_variant(device_id: int) -> Optional[str]:
    """The SoC variant from ``npu-smi info -t board`` ("NPU Name", e.g. 957b),
    used to disambiguate a family chip name (Ascend950PR) into a concrete arch
    token. None when the field is absent or the probe fails."""
    try:
        r = subprocess.run(
            ["npu-smi", "info", "-t", "board", "-i", str(int(device_id))],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    match = re.search(r"NPU\s*Name\s*:\s*(\S+)", r.stdout)
    return match.group(1) if match else None


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


def _derive_arch_cpu(_device_id: int) -> Optional[str]:
    return normalize_cpu_arch_name()


# backend -> (probe fn, user-facing hint shown when arch derivation fails)
_PROBES = {
    "ascend": (_derive_arch_ascend, "is npu-smi on PATH?"),
    "cuda": (_derive_arch_cuda, "is nvidia-smi on PATH?"),
    "cpu": (_derive_arch_cpu, "platform.machine() returned empty"),
}


def derive_arch(device_id: int, backend: str = "ascend") -> Optional[str]:
    """Return the arch token for ``device_id`` on ``backend``.

    Returns None if the probe is unavailable or unparseable; callers decide
    whether that is fatal for their workflow.
    """
    probe = _PROBES.get((backend or "ascend").lower())
    return probe[0](device_id) if probe else None


def probe_hint(backend: str) -> str:
    """User-facing probe-tool hint for arch derivation errors."""
    probe = _PROBES.get((backend or "ascend").lower())
    return probe[1] if probe \
        else f"no arch probe registered for backend={backend!r}"
