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

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512}),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1024}),
    ],
    key=['M', 'N']
)
@triton.jit
def sum_kernel(
    in_ptr0, out_ptr0, 
    in_stride0, in_stride1, 
    out0_stride0, 
    M, 
    N,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    m_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mmask = m_offsets < M

    row_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        block_vals = tl.load(block_ptrs, mask=mask, other=0.0)

        row_sum += block_vals
        
    row_sum = tl.sum(row_sum, axis=1)
    output0_ptrs = out_ptr0 + m_offsets * out0_stride0
    tl.store(output0_ptrs, row_sum, mask=mmask)


def sum_triton_torch(input0):
    """
    2D, reduce_axis = 1
    """
    M, N = input0.shape
    output0 = torch.empty((M,), dtype=input0.dtype, device=input0.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )

    sum_kernel[grid](
        input0, 
        output0, 
        input0.stride(0),
        input0.stride(1),
        output0.stride(0),
        M, 
        N
    )
    return output0