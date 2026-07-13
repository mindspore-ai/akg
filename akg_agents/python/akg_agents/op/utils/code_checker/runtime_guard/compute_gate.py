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

"""Runtime anti-cheat gate: disable core-compute delegation at the dispatch layer.

Wraps the candidate op's correctness run. Instead of *watching* for cheats with a
TorchFunctionMode / TorchDispatchMode (both of which are suspended while a
registered custom op runs, so ATen calls nested inside the candidate's own op are
invisible — the delegation blind spot), this DISABLES the forbidden ops at the
kernel-registration layer via ``torch.library``. A disabled leaf raises no matter
who dispatches it — Python, C++, or nested inside the candidate's custom op — so
the blind spot is closed by construction. Validated on-device (NPU / PrivateUse1):
a candidate that delegates matmul directly or nested in its own custom op raises;
legit elementwise / dtype-cast / small host reads pass.

Overrides are registered per-call on CPU + the NPU (PrivateUse1) key and torn down
with ``Library._destroy()`` when the gate exits, so the next case's golden /
reference (which legitimately uses matmul etc.) is unaffected.

Scope:
- Core-compute leaves (mm / bmm / addmm / conv* / einsum / topk / sort / _softmax
  / native_layer_norm ...): raised on any dispatch, on BOTH keys. The CPU-key
  override also covers "move inputs to host, torch.matmul there, copy back" — the
  host matmul dispatches aten::mm on CPU and raises.
- NOT gated: bulk D2H by itself. A kernel-layer ``_to_copy`` override is not viable
  on NPU (the device transfer needs the PrivateUse1 kernel that a recursion-guard
  exclude would remove; it would also break legit on-device dtype casts). So a
  candidate reconstructing the result on host purely from primitives / numpy
  (never naming a blocked leaf) is not caught here — an exotic path that the perf
  stage rejects anyway, and one the old watcher never robustly closed either.
- Raw ``aclnn*`` ACL calls and ``torch_npu.npu_*`` builtins never reach ATen
  dispatch; the static CodeChecker scan blocks those at the source level."""

from __future__ import annotations

import contextlib

import torch

from ._policy import COMPUTE_LEAVES

__all__ = ["compute_gate", "BuiltinComputeError"]

# Dispatch keys the candidate's tensors live on: CPU (host-delegation attempt)
# and the NPU backend (torch_npu registers under PrivateUse1).
_GATE_KEYS = ("CPU", "PrivateUse1")


class BuiltinComputeError(RuntimeError):
    """Candidate delegated core compute to a builtin (raised in enforce mode)."""


@contextlib.contextmanager
def compute_gate(mode: str, leaves=None):
    """Disable the core-compute leaves for the wrapped candidate run.

    mode: ``off`` installs nothing; anything else enforces (a hit fails the case).
    A disabled leaf raises unconditionally — there is no reliable "detect but let
    through" for a leaf (redispatching past the override leaves no backend kernel),
    and enforce is the only mode that matters.
    """
    if mode == "off":
        yield
        return
    leaves = tuple(leaves if leaves is not None else COMPUTE_LEAVES)

    def _leaf_override(name):
        def _fn(*args, **kwargs):
            raise BuiltinComputeError(
                f"[SECURITY] builtin compute violation: candidate dispatched "
                f"aten::{name} (Python, C++ torch::{name}, or nested in its own "
                f"custom op). An AscendC kernel must compute the core itself, "
                f"not delegate matmul/conv/topk/softmax to a builtin.")
        return _fn

    lib = torch.library.Library("aten", "IMPL")
    for key in _GATE_KEYS:
        for name in leaves:
            try:
                lib.impl(name, _leaf_override(name), key)
            except Exception:
                pass  # leaf absent for this key / build — nothing to disable
    try:
        yield
    finally:
        lib._destroy()  # restore original kernels for the next case / golden
