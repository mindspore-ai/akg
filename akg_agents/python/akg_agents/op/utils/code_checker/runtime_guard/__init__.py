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

"""Runtime anti-cheat — the execution-time half of the CodeChecker.

``op.utils.code_checker`` does the *static* anti-cheat (source-regex scan driven
by ``config/code_checker.yaml``, blocking the raw ``aclnn*`` / ``torch_npu.npu_*``
calls that never reach ATen dispatch). This submodule does the *runtime* half from
the same config (``ascendc_anti_cheat.runtime_guard``): during the candidate's
correctness run, :func:`compute_gate` disables the core-compute leaves and bulk
D2H egress at the dispatch layer, so delegation via Python, C++ ``torch::*``, or
nesting inside the candidate's own custom op all raise — closed by construction,
no watcher blind spot."""

from __future__ import annotations

import os
from typing import Callable, Optional, TypeVar

from ._policy import DEFAULT_MODE as _DEFAULT_MODE
from .compute_gate import compute_gate, BuiltinComputeError

__all__ = [
    "compute_gate",
    "BuiltinComputeError",
    "guard_mode",
    "guarded_call",
]

_VALID_MODES = ("block", "warn", "off")

T = TypeVar("T")


def guard_mode() -> str:
    """Resolve the runtime guard mode: ``AKG_GUARD_MODE`` env, else config default."""
    mode = os.environ.get("AKG_GUARD_MODE", _DEFAULT_MODE).strip().lower()
    return mode if mode in _VALID_MODES else _DEFAULT_MODE


def guarded_call(fn: Callable[[], T], mode: Optional[str] = None) -> T:
    """Run the *candidate* op execution ``fn()`` under :func:`compute_gate` and
    return its result. Emitted as a single expression by the AscendC verify
    adapter, so it stays indentation-neutral in the generated verify script."""
    with compute_gate(mode or guard_mode()):
        return fn()
