# Copyright 2025 Huawei Technologies Co., Ltd
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

import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,  # 输入指针
    output_ptr,  # 输出指针
    n_elements,  # 总元素数
    BLOCK_SIZE: tl.constexpr,  # 每个block处理的元素数
):
    # 获取程序ID
    pid = tl.program_id(axis=0)
    # 计算这个block的起始位置
    block_start = pid * BLOCK_SIZE
    # 创建偏移量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码，确保不越界
    mask = offsets < n_elements

    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 执行ReLU: max(0, x)
    output = tl.maximum(x, 0.0)

    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)


def relu_triton_mindspore(x):
    import mindspore as ms
    output = ms.mint.zeros_like(x)

    n_elements = output.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    relu_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def relu_triton_torch(x):
    x = x.contiguous()
    n_elements = x.numel()
    output = torch.empty_like(x, device=x.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # 启动kernel
    relu_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
