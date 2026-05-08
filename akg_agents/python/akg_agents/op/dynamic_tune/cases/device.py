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

from __future__ import annotations

import contextlib
from typing import Any, Iterator


def _device_id_from(device: str) -> int:
    try:
        return int(str(device).split(":", 1)[1])
    except Exception:
        return 0


def _set_npu_device(device: str) -> None:
    try:
        import torch_npu  # type: ignore
    except ImportError:
        return
    torch_npu.npu.set_device(_device_id_from(device))


@contextlib.contextmanager
def _push_default_device(device: str | None) -> Iterator[bool]:
    """临时把 torch 的 default device 切到 ``device``，让 ``base.get_inputs_dyn_list``
    里那些不带 ``device=`` 的 ``torch.randn`` / ``torch.zeros`` / ``torch.full`` 直接
    落到 NPU 上，省一次 H2D。

    ``device=None`` 时 no-op；老版 torch (<2.0) 没有 ``set_default_device`` 时也 no-op，
    调用方应保留 ``.to(device)`` 兜底。
    """
    import torch  # type: ignore

    if device is None or not hasattr(torch, "set_default_device"):
        yield False
        return
    torch.set_default_device(device)
    try:
        yield True
    finally:
        torch.set_default_device("cpu")


def _maybe_empty_npu_cache(torch_mod: Any) -> None:
    """每跑完一个 case 调一下，把上一 shape 的 outputs / 中间 tensor 还给 HBM。

    没 ``torch.npu.empty_cache`` (例如非 NPU build) 时静默 noop。
    """
    npu_mod = getattr(torch_mod, "npu", None)
    if npu_mod is None:
        return
    empty_cache = getattr(npu_mod, "empty_cache", None)
    if empty_cache is None:
        return
    try:
        empty_cache()
    except Exception:
        pass
