# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   chunked_sgmv_lora_shrink_forward(x, weights, batch_info, num_slices)
# Triton Kernel:
#   _chunked_lora_shrink_kernel - 这个内核函数主要实现计算LoRA Shrink矩阵的乘积。
# ============================================================================
from typing import Optional

import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_shrink_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Information on sequence lengths,ranks and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # Meta parameters
    N: tl.constexpr,  # num_slices * r
    K: tl.constexpr,  # input_dim
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes a chunked SGMV for LoRA shrink operations.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * num_slices]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        weights (torch.Tensor): The LoRA A weights for all available adapters,
            with shape `(num_lora, N, K)` where N = num_slices * r.
        output (torch.Tensor): The output tensor of shape `(s, N)`.
    """
    x_stride_1: tl.constexpr = 1
    x_stride_0: tl.constexpr = K

    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return

    pid_n = tl.program_id(0)

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)

    # Adjust N dim according to the specific LoRA adapter
    cur_n = tl.minimum(N, rank * NUM_SLICES)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = x + (
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_lora_shrink_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    use_cuda_graph: bool,
    bs: int,
    num_segments: int,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    scalings: torch.Tensor,
    max_len: Optional[int],
    seg_lens: Optional[torch.Tensor],
    permutation: Optional[torch.Tensor],
    num_slices: int,
) -> torch.Tensor:
    # x: (s, input_dim)
    # weights: (num_lora, num_slices * r, input_dim)
    # output: (s, num_slices * r)
    # num_slices: qkv=3, gate_up=2, others=1
    # when called with multiple slices, the weights.shape[-2] will be num_slices * r
    # input_dim is much larger than r

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    # Block shapes
    # TODO (lifuhuang): experiment with split-k
    BLOCK_M = max_len
    BLOCK_N = 16
    BLOCK_K = 256

    S = x.shape[0]
    N = weights.shape[1]
    K = weights.shape[2]
    assert x.shape[-1] == K

    grid = (
        triton.cdiv(N, BLOCK_N),
        bs if use_cuda_graph else num_segments,
    )

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    _chunked_lora_shrink_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        permutation=permutation,
        num_segs=num_segments,
        # constants
        N=N,
        K=K,
        NUM_SLICES=num_slices,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(
        self,
        x,
        weights,
        use_cuda_graph,
        bs,
        num_segments,
        seg_indptr,
        weight_indices,
        lora_ranks,
        scalings,
        max_len,
        seg_lens,
        permutation,
        num_slices=3,
    ):
        return chunked_sgmv_lora_shrink_forward(
            x,
            weights,
            use_cuda_graph,
            bs,
            num_segments,
            seg_indptr,
            weight_indices,
            lora_ranks,
            scalings,
            max_len,
            seg_lens,
            permutation,
            num_slices,
        )


class ModelSglang(nn.Module):
    def __init__(self):
        super(ModelSglang, self).__init__()
    
    def forward(
        self,
        x,
        weights,
        use_cuda_graph,
        bs,
        num_segments,
        seg_indptr,
        weight_indices,
        lora_ranks,
        scalings,
        max_len,
        seg_lens,
        permutation,
        num_slices=3,
    ):
        from sglang.srt.lora.triton_ops.chunked_sgmv_shrink import (
            chunked_sgmv_lora_shrink_forward,
        )
        # Note: SGLang's function still uses LoRABatchInfo, so we need to reconstruct it
        from sglang.srt.lora.utils import LoRABatchInfo
        batch_info = LoRABatchInfo(
            use_cuda_graph=use_cuda_graph,
            bs=bs,
            num_segments=num_segments,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            max_len=max_len,
            seg_lens=seg_lens,
            permutation=permutation,
        )
        return chunked_sgmv_lora_shrink_forward(x, weights, batch_info, num_slices)


def get_inputs():
    batch_size = 4
    max_seq_len = 32
    num_slices = 3  # QKV
    max_rank = 64  # 最大 LoRA rank
    input_dim = 4096  # 输入维度
    
    # 生成序列长度
    seq_lengths = [max_seq_len] * batch_size
    total_seq_len = sum(seq_lengths)
    
    # x: (total_seq_len, input_dim) - 输入激活
    x = torch.randn(
        total_seq_len, input_dim, dtype=torch.float16
    )
    
    # weights: (num_lora, num_slices * max_rank, input_dim) - LoRA A weights
    # 假设有 2 个 LoRA adapters
    num_loras = 2
    weights = torch.randn(
        num_loras, num_slices * max_rank, input_dim, dtype=torch.float16
    )
    
    num_segments = batch_size
    seg_indptr = torch.zeros(num_segments + 1, dtype=torch.int32)
    for i in range(1, num_segments + 1):
        seg_indptr[i] = seg_indptr[i - 1] + seq_lengths[i - 1]
    
    weight_indices = torch.zeros(num_segments, dtype=torch.int32)  # 都使用第一个 LoRA
    lora_ranks = torch.tensor([max_rank, 32], dtype=torch.int32)  # 两个 LoRA 的 rank
    scalings = torch.ones(num_loras, dtype=torch.float32)  # 缩放因子
    
    permutation = torch.arange(total_seq_len, dtype=torch.int32)
    
    use_cuda_graph = False
    bs = batch_size
    max_len = max_seq_len
    seg_lens = torch.tensor(seq_lengths, dtype=torch.int32)
    
    return [
        x,
        weights,
        use_cuda_graph,
        bs,
        num_segments,
        seg_indptr,
        weight_indices,
        lora_ranks,
        scalings,
        max_len,
        seg_lens,
        permutation,
        num_slices,
    ]


def get_init_inputs():
    return []