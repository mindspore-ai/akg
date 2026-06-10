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

"""Adapter factories — single source of truth for DSL / framework /
backend adapter registration AND the per-family DSL whitelist.

The historical pair (this factory's if/elif + config_utils._DSL_TABLE)
drifted: a new DSL had to be added in two places. Now ``DSL_REGISTRY``
is the only place that knows about a DSL — config_utils derives its
support matrix from it.

Each :class:`DSLEntry` carries:
  * ``module`` / ``cls`` — lazy adapter import (factory call is the only
    place the DSL's module is touched, keeps import-time cost down).
  * ``aliases`` — equivalent names accepted at lookup (e.g.
    "triton-russia" → DSLAdapterTritonAscend).
  * ``support`` — tuples ``(framework, backend, family)`` declaring where
    this DSL is allowed; config_utils._DSL_TABLE is rebuilt from these.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class DSLEntry:
    module: str
    cls: str
    aliases: Tuple[str, ...] = ()
    support: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# DSL_REGISTRY — single source of truth. Adding a new DSL = one entry.
# ---------------------------------------------------------------------------
# family tag is internal to config_utils._family_of; valid values are
# "910" / "310" for ascend, "any" for cuda / cpu. Keep framework
# dimensions explicit: a DSL can support torch/910 without supporting
# mindspore/910.
_TORCH_ASCEND_910 = (("torch", "ascend", "910"),)
_TORCH_ASCEND_310 = (("torch", "ascend", "310"),)
_MINDSPORE_ASCEND_910 = (("mindspore", "ascend", "910"),)
_MINDSPORE_ASCEND_310 = (("mindspore", "ascend", "310"),)
_NUMPY_ASCEND_310 = (("numpy", "ascend", "310"),)
_TORCH_CUDA = (("torch", "cuda", "any"),)
_TORCH_CPU = (("torch", "cpu", "any"),)

DSL_REGISTRY: dict = {
    "triton_cuda": DSLEntry(
        module="triton_cuda", cls="DSLAdapterTritonCuda",
        support=_TORCH_CUDA,
    ),
    "triton_ascend": DSLEntry(
        module="triton_ascend", cls="DSLAdapterTritonAscend",
        aliases=("triton-russia",),
        support=_TORCH_ASCEND_910 + _MINDSPORE_ASCEND_910,
    ),
    "swft": DSLEntry(
        module="swft", cls="DSLAdapterSwft",
        support=(
            _TORCH_ASCEND_310
            + _MINDSPORE_ASCEND_310
            + _NUMPY_ASCEND_310
        ),
    ),
    "ascendc": DSLEntry(
        module="ascendc", cls="DSLAdapterAscendC",
        support=_TORCH_ASCEND_910 + _TORCH_ASCEND_310,
    ),
    "ascendc_catlass": DSLEntry(
        module="ascendc_catlass", cls="DSLAdapterAscendC_Catlass",
        support=_TORCH_ASCEND_910 + _TORCH_ASCEND_310,
    ),
    "cpp": DSLEntry(
        module="cpp", cls="DSLAdapterCpp",
        support=_TORCH_CPU,
    ),
    "cuda_c": DSLEntry(
        module="cuda_c", cls="DSLAdapterCudaC",
        support=_TORCH_CUDA,
    ),
    "tilelang_npuir": DSLEntry(
        module="tilelang_npuir", cls="DSLAdapterTilelangNpuir",
        support=_TORCH_ASCEND_910,
    ),
    "tilelang_ascend": DSLEntry(
        module="tilelang_ascend", cls="DSLAdapterTilelangAscend",
        support=_TORCH_ASCEND_910,
    ),
    "tilelang_cuda": DSLEntry(
        module="tilelang_cuda", cls="DSLAdapterTilelangCuda",
        support=_TORCH_CUDA,
    ),
    "torch": DSLEntry(
        module="torch", cls="DSLAdapterTorch",
        support=_TORCH_ASCEND_910 + _TORCH_ASCEND_310 + _TORCH_CUDA,
    ),
    "pypto": DSLEntry(
        module="pypto", cls="DSLAdapterPypto",
        support=_TORCH_ASCEND_910,
    ),
}

# Alias → canonical lookup, built once.
_DSL_ALIAS_MAP: dict = {}
for _name, _entry in DSL_REGISTRY.items():
    _DSL_ALIAS_MAP[_name.lower()] = _name
    for _alias in _entry.aliases:
        _DSL_ALIAS_MAP[_alias.lower()] = _name


def _resolve_dsl_name(dsl: str) -> str:
    canonical = _DSL_ALIAS_MAP.get(dsl.lower())
    if canonical is None:
        raise ValueError(f"Unsupported DSL: {dsl}")
    return canonical


def get_dsl_adapter(dsl: str):
    """Get DSL adapter by name (or alias)."""
    entry = DSL_REGISTRY[_resolve_dsl_name(dsl)]
    module = __import__(
        f"akg_agents.op.verifier.adapters.dsl.{entry.module}",
        fromlist=[entry.cls],
    )
    return getattr(module, entry.cls)()


def get_framework_adapter(framework: str):
    """Get framework adapter by name (torch / mindspore / numpy)."""
    framework_lower = framework.lower()
    if framework_lower == "torch":
        from .framework.torch import FrameworkAdapterTorch
        return FrameworkAdapterTorch()
    if framework_lower == "mindspore":
        from .framework.mindspore import FrameworkAdapterMindSpore
        return FrameworkAdapterMindSpore()
    if framework_lower == "numpy":
        from .framework.numpy import FrameworkAdapterNumpy
        return FrameworkAdapterNumpy()
    raise ValueError(f"Unsupported framework: {framework}")


def get_backend_adapter(backend: str):
    """Get backend adapter by name (cuda / ascend / cpu)."""
    backend_lower = backend.lower()
    if backend_lower == "cuda":
        from .backend.cuda import BackendAdapterCuda
        return BackendAdapterCuda()
    if backend_lower == "ascend":
        from .backend.ascend import BackendAdapterAscend
        return BackendAdapterAscend()
    if backend_lower == "cpu":
        from .backend.cpu import BackendAdapterCpu
        return BackendAdapterCpu()
    raise ValueError(f"Unsupported backend: {backend}")
