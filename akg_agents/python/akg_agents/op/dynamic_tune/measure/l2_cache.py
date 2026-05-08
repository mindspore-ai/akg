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

"""L2 cache clear 实用函数。

调优阶段每个被测 launch 之间需要把 NPU L2 cache 清掉，避免上一次 launch 把
hot data 留在 L2 给下一次 launch 蹭热度，污染 latency。

实现采用与上游同名约定的 `AKG_l2cache_clear` triton kernel：写 0 到一块大于
L2 容量的连续 buffer。lazy 加载 torch / triton，让纯逻辑测试不需要这两个依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_L2_CACHE_BYTES = 192 * 1024 * 1024
DEFAULT_NUM_PROGRAMS = 32   # ascend 910B 一个 SoC 上的 vector core 数；固定值最稳。
DEFAULT_BLOCK_SIZE = 32768  # 单个 program 单次处理的元素数；与上面成对，避免每次重编译。

_BUFFER_CACHE: dict[tuple[str, int, int], Any] = {}


@dataclass(frozen=True)
class L2CacheClearer:
    """绑定到具体 device 的 L2 cache clear 句柄。

    实现思路：
        - 固定 grid=NUM_PROGRAMS（与 ascend vector core 数一致），不用 cdiv 启
          很多 program；
        - kernel 内部走 persistent loop，把 cache_bytes 大小的 buffer 写一遍 0；
        - kernel 名字必须是 'AKG_l2cache_clear'，下游解析模块要拿它做 step 边界。
    """

    device: Any
    cache_bytes: int = DEFAULT_L2_CACHE_BYTES
    num_programs: int = DEFAULT_NUM_PROGRAMS
    block_size: int = DEFAULT_BLOCK_SIZE

    def clear(self) -> None:
        # lazy import：让无 triton 的纯逻辑测试能 import 上层模块
        import torch  # type: ignore
        from akg_agents.op.dynamic_tune.measure._npu_kernels import AKG_l2cache_clear

        buffer = _get_clear_buffer(self.device, self.cache_bytes)
        # Triton/NPU launch 依赖当前 device 上下文；仅靠 tensor.device 不足以保证
        # kernel 真正在目标卡上发射。这里显式切换到绑定 device，避免误落到默认卡。
        with torch.npu.device(self.device):  # type: ignore[attr-defined]
            AKG_l2cache_clear[(int(self.num_programs),)](
                buffer, buffer.numel(), int(self.num_programs), int(self.block_size)
            )


def _get_clear_buffer(device: Any, cache_bytes: int) -> Any:
    import torch  # type: ignore

    device_key = str(device)
    key = (device_key, int(cache_bytes), torch.float32.itemsize)
    if key in _BUFFER_CACHE:
        return _BUFFER_CACHE[key]
    n_elements = max(1, int(cache_bytes) // torch.float32.itemsize)
    buffer = torch.empty(n_elements, dtype=torch.float32, device=device)
    _BUFFER_CACHE[key] = buffer
    return buffer


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_L2_CACHE_BYTES",
    "DEFAULT_NUM_PROGRAMS",
    "L2CacheClearer",
]
