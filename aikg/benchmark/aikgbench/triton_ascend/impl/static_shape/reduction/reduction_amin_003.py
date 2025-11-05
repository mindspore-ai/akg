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
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048}),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096}),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 8192}),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16384}),
    ],
    key=['M', 'N']
)
@triton.jit
def amin_kernel(
    in_ptr0, out_ptr0, 
    in_stride0, in_stride1, 
    out_stride0, 
    M, 
    N,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    m_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mmask = m_offsets < M

    curr_min = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        data_block = tl.load(block_ptrs, mask=mask, other=float('inf'))
        curr_min = tl.minimum(data_block, curr_min)
    row_min = tl.min(curr_min, 1)

    output_ptrs = out_ptr0 + m_offsets * out_stride0
    tl.store(output_ptrs, row_min, mask=mmask)

def amin_triton_torch(input0):
    """
    2D, reduce_axis = 1
    """
    M, N = input0.shape
    output0 = torch.empty((M,), dtype=input0.dtype, device=input0.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )

    amin_kernel[grid](
        input0, 
        output0, 
        input0.stride(0),
        input0.stride(1),
        output0.stride(0),
        M, 
        N
    )
    return output0