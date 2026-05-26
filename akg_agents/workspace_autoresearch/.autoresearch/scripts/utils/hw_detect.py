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

"""Hardware detection + DSL→backend mapping.

User picks `--dsl` + (`--devices N` or `--worker-url URL`); we derive
backend from a static DSL table and arch from the hardware.

Two resolution paths:

    local:   --devices N  →  npu-smi / nvidia-smi / uname -m  →  arch
    remote:  --worker-url →  GET /api/v1/status              →  {backend, arch, devices}

Stdlib + subprocess + urllib only — safe to import from anywhere.
"""

# pylint: disable=broad-exception-caught,missing-function-docstring
from __future__ import annotations

import json
import re
import subprocess
from typing import Optional
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# DSL → backend (static table; DSL name encodes target backend)
# ---------------------------------------------------------------------------

_DSL_BACKEND = {
    "triton_ascend":   "ascend",
    "triton_cuda":     "cuda",
    "ascendc":         "ascend",
    "cuda_c":          "cuda",
    "tilelang_cuda":   "cuda",
    "tilelang_npuir":  "ascend",
    "pypto":           "ascend",
    "swft":            "ascend",
    "cpp":             "cpu",
    "torch":           "cpu",
}


def backend_for_dsl(dsl: str) -> str:
    key = dsl.lower()
    if key not in _DSL_BACKEND:
        raise ValueError(f"Unknown DSL {dsl!r}; known: {sorted(_DSL_BACKEND)}")
    return _DSL_BACKEND[key]


def list_supported_dsls() -> tuple:
    """Sorted tuple of all DSL names. Single source of truth for the DSL
    menu surfaced in LLM-facing text (scaffold --help, parse_args
    missing-fields payload, slash-command docs)."""
    return tuple(sorted(_DSL_BACKEND))


_BACKEND_DEVICE_TYPE = {
    "ascend": "npu",    # torch.device("npu:N") via torch_npu
    "cuda":   "cuda",
    "cpu":    "cpu",
}


def device_type_for_backend(backend: str) -> str:
    key = backend.lower()
    if key not in _BACKEND_DEVICE_TYPE:
        raise ValueError(f"Unknown backend {backend!r}")
    return _BACKEND_DEVICE_TYPE[key]


def device_type_for_dsl(dsl: str) -> str:
    return device_type_for_backend(backend_for_dsl(dsl))


# ---------------------------------------------------------------------------
# Arch derivation from local hardware
# ---------------------------------------------------------------------------

def derive_arch(backend: str, device_id: int) -> Optional[str]:
    """Return arch string (e.g. 'ascend910b3', 'a100', 'x86_64') or None
    if detection fails. Caller decides whether None is fatal."""
    backend = backend.lower()
    if backend == "ascend":
        return _npu_arch(device_id)
    if backend == "cuda":
        return _cuda_arch(device_id)
    if backend == "cpu":
        return _cpu_arch()
    return None


def _npu_arch(device_id: int) -> Optional[str]:
    """Parse `npu-smi info` main table for the NPU's Name column.

    The table row looks like:
        | 5     910B3               | Alarm         | ...
    We match the leading `<device_id>` and capture the next token →
    'ascend910b3'. `npu-smi info -t board -i N` exposes Product/Model but
    not the architecture string, so we go through the main table instead.
    """
    try:
        r = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    pat = re.compile(rf"^\|\s*{int(device_id)}\s+(\S+)\s*\|", re.MULTILINE)
    m = pat.search(r.stdout)
    if not m:
        return None
    name = m.group(1).strip().lower()
    # Names come back as '910b3', '910b4', '910b2', '310p3', etc. — prefix
    # with 'ascend' to match the ROOFLINE_ARCH_CONFIGS keys.
    return f"ascend{name}"


def _cuda_arch(device_id: int) -> Optional[str]:
    """nvidia-smi → common arch shorthand (a100 / h100 / etc.).
    Fallback to the full name if we don't recognize it."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             "-i", str(device_id)],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    name = r.stdout.strip().lower()
    for token in ("a100", "h100", "a800", "h800", "v100", "t4",
                  "rtx4090", "rtx3090", "l40", "l4"):
        if token in name.replace(" ", "").replace("-", ""):
            return token
    return name or None


def _cpu_arch() -> Optional[str]:
    try:
        r = subprocess.run(["uname", "-m"], capture_output=True, text=True,
                           timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None


# ---------------------------------------------------------------------------
# Worker status fetch (remote path)
# ---------------------------------------------------------------------------

def fetch_worker_hardware(worker_url: str, timeout: float = 5.0) -> Optional[dict]:
    """GET /api/v1/status on the worker. Returns a dict like
    {"status": "ready", "backend": "ascend", "arch": "ascend910b3",
     "devices": [5]} or None on failure.
    """
    url = worker_url.strip()
    if not url.startswith("http"):
        url = f"http://{url}"
    url = url.rstrip("/") + "/api/v1/status"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data
