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

"""真实 NPU 路径用到的 triton kernel 集合。

单独成文件的原因：triton.jit 通过 AST 解析 kernel 源码，需要 `triton` /
`triton.language as tl` 在 kernel 函数所在模块的顶层 globals 里能找到。
如果把 @triton.jit 函数定义在某个 `def` 函数体内，AST 解析时就拿不到 `tl`。

本文件**顶层** import triton 与 triton.language——单元测试请勿 import 这个模块，
它依赖真实 triton 安装；NpuProfilerBackend / L2CacheClearer 在调用时才会 lazy
import 它。
"""

from __future__ import annotations

import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def AKG_l2cache_clear(  # noqa: N802
    output_ptr,
    n_elements,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ascend 标准 persistent loop：固定 grid=NUM_PROGRAMS，每个 program
    在 buffer 上交错循环处理 BLOCK_SIZE 元素，确保整个 buffer 都被写一遍。

    用 cdiv 启动巨多 program 在 ascend 上不友好（很容易触发 MTE out of range 与
    vector core exception），固定核心数 + persistent loop 才是 ascend 推荐模式。

    选这个名字是因为下游解析模块要靠 'AKG_l2cache_clear' 这个串识别 L2 clear
    边界，**不要重命名**。
    """

    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for blk in range(pid, num_blocks, NUM_PROGRAMS):
        offsets = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, 0.0, mask=mask)


L2_CACHE_CLEAR_KERNEL_NAME = "AKG_l2cache_clear"

__all__ = ["AKG_l2cache_clear", "L2_CACHE_CLEAR_KERNEL_NAME"]
