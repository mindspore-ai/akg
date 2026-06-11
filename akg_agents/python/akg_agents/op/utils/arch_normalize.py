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

"""Backend arch-name normalization helpers.

CUDA model names are open-ended, so normalization is family-pattern based
instead of a concrete SKU table. The goal is to turn probe output into the
short arch token used by AKG while keeping future cards data-driven.
"""

from __future__ import annotations

import platform
import re
from typing import Optional


ASCEND_ARCH_PATTERN = re.compile(r"^ascend[0-9][0-9a-z]*(?:_[0-9a-z]+)?$")
CUDA_ARCH_PATTERN = re.compile(
    r"^(?:rtx\d{3,4}[a-z]?|gtx\d{3,4}[a-z]?|[ahvltb]\d{1,4}[a-z]?)$"
)
CPU_ARCH_PATTERN = re.compile(r"^(?:x86_64|aarch64|riscv64|ppc64le)$")

_CUDA_MARKETING_NOISE = re.compile(
    r"\b(?:nvidia|tesla|geforce|quadro|titan|laptop|gpu|pcie|sxm\d*|hbm\d?|"
    r"\d+\s*gb)\b",
    re.IGNORECASE,
)
_CUDA_MODEL_PATTERN = re.compile(
    r"\b(rtx[\s\-]*\d{3,4}[a-z]?|gtx[\s\-]*\d{3,4}[a-z]?|"
    r"[ahvltb]\d{1,4}[a-z]?)\b",
    re.IGNORECASE,
)


def normalize_ascend_arch_name(name: str) -> Optional[str]:
    """Return AKG's canonical Ascend arch token from a probe chip name.

    ``npu-smi info`` usually exposes a chip token such as ``910B3`` or
    ``950PR_9589``. Normalize only the spelling here; support policy for
    concrete SKUs remains in config validation.
    """
    if not isinstance(name, str):
        return None
    token = re.sub(r"[\s\-]+", "", name.strip().lower())
    if not token:
        return None
    arch = token if token.startswith("ascend") else f"ascend{token}"
    return arch if ASCEND_ARCH_PATTERN.match(arch) else None


def normalize_cuda_arch_name(name: str) -> Optional[str]:
    """Return AKG's short CUDA arch token from a vendor model string.

    This intentionally does not enumerate known GPU SKUs. It strips common
    marketing words from the probe result, extracts a family-shaped token,
    lowercases it, and removes interior whitespace/dashes.
    """
    if not isinstance(name, str):
        return None
    cleaned = _CUDA_MARKETING_NOISE.sub(" ", name)
    match = _CUDA_MODEL_PATTERN.search(cleaned)
    if not match:
        return None
    token = re.sub(r"[\s\-]+", "", match.group(1).lower())
    return token if CUDA_ARCH_PATTERN.match(token) else None


def normalize_cpu_arch_name(name: Optional[str] = None) -> Optional[str]:
    """Normalize platform CPU arch names into AKG's canonical tokens."""
    raw = name if name is not None else platform.machine()
    arch = (raw or "").strip().lower()
    if arch in ("x86_64", "amd64"):
        return "x86_64"
    if arch in ("aarch64", "arm64"):
        return "aarch64"
    return arch if CPU_ARCH_PATTERN.match(arch) else None


def normalize_arch_name(backend: str, name: Optional[str] = None) -> Optional[str]:
    """Dispatch to the backend-specific arch normalizer."""
    b = (backend or "").strip().lower()
    if b in ("ascend", "npu"):
        return normalize_ascend_arch_name(name or "")
    if b == "cuda":
        return normalize_cuda_arch_name(name or "")
    if b == "cpu":
        return normalize_cpu_arch_name(name)
    return None


def ascend_soc_version(arch: str) -> Optional[str]:
    """Derive the CANN/AscendC ``SOC_VERSION`` token from an AKG arch.

    Keep support policy outside this function: config validation decides
    which concrete SKUs are accepted. This function only encodes naming
    rules needed by AscendC's ``run.sh -v``.
    """
    normalized = normalize_ascend_arch_name(arch or "")
    if not normalized:
        return None
    suffix = normalized[len("ascend"):]

    match = re.fullmatch(r"(910b[0-9][a-z]?)", suffix)
    if match:
        return "Ascend" + match.group(1).upper()

    match = re.fullmatch(r"(310p[0-9][a-z]?)", suffix)
    if match:
        return "Ascend" + match.group(1).upper()

    match = re.fullmatch(r"910_([0-9a-z]+)", suffix)
    if match:
        return f"Ascend910_{match.group(1)}"

    match = re.fullmatch(r"950dt_([0-9a-z]+)", suffix)
    if match:
        return f"Ascend950DT_{match.group(1).upper()}"

    match = re.fullmatch(r"950pr_([0-9a-z]+)", suffix)
    if match:
        return f"Ascend910_{match.group(1)}"

    return None


def ascend_direct_invoke_npu_arch(arch: str) -> Optional[str]:
    """Derive CANN direct-invoke ``--npu-arch`` from an AKG Ascend arch.

    CANNBot-style direct-invoke projects compile ASC kernels with dav
    architecture tokens. Keep this rule prefix-based so new SKUs admitted by
    config validation do not require a second hard-coded table here.
    """
    normalized = normalize_ascend_arch_name(arch or "")
    if not normalized:
        return None
    if normalized.startswith("ascend950"):
        return "dav-3510"
    if normalized.startswith("ascend910") or normalized.startswith("ascend310"):
        return "dav-2201"
    return None
