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

import functools
import glob
import os
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


@functools.lru_cache(maxsize=1)
def load_ascend_soc_catalog() -> frozenset:
    """Valid SOC_VERSION strings (exact case) read from CANN's
    ``platform_config`` — the source of truth for both spelling and case.

    Scans ``<ASCEND_HOME>/**/platform_config/**/*.ini`` once (cached) and
    returns the file stems. Empty when CANN / the env var is absent (dev
    hosts, CI), so callers degrade gracefully and tests inject their own.
    """
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_HOME")
    if not home or not os.path.isdir(home):
        return frozenset()
    try:
        pattern = os.path.join(home, "**", "platform_config", "**", "*.ini")
        return frozenset(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.iglob(pattern, recursive=True)
        )
    except OSError:
        return frozenset()


def ascend_soc_version(arch: str,
                       catalog: Optional[frozenset] = None) -> Optional[str]:
    """CANN/AscendC ``SOC_VERSION`` for an AKG ascend arch token.

    SOC_VERSION is the chip name verbatim, so ``arch_token == soc.lower()``.
    Instead of hand-reconstructing the (case-sensitive, per-family-inconsistent)
    spelling, look the token up in CANN's own ``platform_config`` catalog
    case-insensitively and return its exact entry — correct for every current
    SoC and any future one CANN ships, with no per-family casing rules.

    ``catalog`` defaults to :func:`load_ascend_soc_catalog`; pass an explicit
    set offline / in tests. Returns ``None`` when the token isn't a real SOC —
    also how :mod:`hw_detect` learns that a bare family name (``ascend950pr``)
    needs its board variant appended before it resolves.
    """
    token = normalize_ascend_arch_name(arch or "")
    if not token:
        return None
    if catalog is None:
        catalog = load_ascend_soc_catalog()
    for soc in catalog:
        if soc.lower() == token:
            return soc
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
