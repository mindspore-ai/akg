# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   chunked_sgmv_lora_expand_forward(x, weights, batch_info, slice_offsets, max_slice_size, base_output)
# Triton Kernel:
#   _chunked_lora_expand_kernel - 这个内核函数主要实现计算LoRA Expand矩阵的乘积。
# ============================================================================
from typing import Optional

import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_expand_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Information on sequence lengths and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # For fused output scaling
    scalings,
    # Offsets of q/k/v slice on output dimension
    slice_offsets,
    # Meta parameters
    NUM_SLICES: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    MAX_RANK: tl.constexpr,  # K = R
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes a chunked SGMV for LoRA expand operations.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, num_slices * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank.
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, output_dim, K).
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, output_dim).
    """
    tl.static_assert(NUM_SLICES <= 3)

    x_stride_0: tl.constexpr = NUM_SLICES * MAX_RANK
    x_stride_1: tl.constexpr = 1

    w_stride_0: tl.constexpr = OUTPUT_DIM * MAX_RANK
    w_stride_1: tl.constexpr = MAX_RANK
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = OUTPUT_DIM
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(axis=2)
    if pid_s >= num_segs:
        return

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v)
    w_index = tl.load(weight_indices + pid_s)
    cur_rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)

    slice_id = tl.program_id(axis=1)
    slice_start = tl.load(slice_offsets + slice_id)
    slice_end = tl.load(slice_offsets + slice_id + 1)

    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter
    cur_rank = tl.minimum(MAX_RANK, cur_rank)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    pid_n = tl.program_id(axis=0)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_start
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = (
        x
        + slice_id * cur_rank * x_stride_1
        + (s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(cur_rank, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < cur_rank - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < cur_rank - k * BLOCK_K)
            & (n_offset[None, :] < slice_end),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (
        n_offset[None, :] < slice_end
    )
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_lora_expand_forward(
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
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    base_output: Optional[torch.Tensor],
) -> torch.Tensor:

    # x: (s, slice_num * r)
    # weights: (num_lora, output_dim, r)
    # slice_offsets: boundaries for different slices in the output dimension
    # output: (s, output_dim)

    # Compute lora_output with shape (s, output_dim) as follows:
    # For each slice i, accumulates:
    # lora_output[:, slice_offsets[i]:slice_offsets[i+1]] += scaling * sgemm(x[:, i*cur_rank:(i+1)*cur_rank], weights[:, slice_offsets[i]:slice_offsets[i+1], :])

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    # Get dims
    M = x.shape[0]
    input_dim = x.shape[1]
    OUTPUT_DIM = weights.shape[1]
    MAX_RANK = weights.shape[2]
    num_slices = len(slice_offsets) - 1
    assert input_dim == num_slices * MAX_RANK

    # TODO (lifuhuang): fine-tune per operation
    BLOCK_M = max_len if max_len is not None else 16
    BLOCK_K = 16
    BLOCK_N = 64

    grid = (
        triton.cdiv(max_slice_size, BLOCK_N),
        num_slices,  # number of slices in the input/output
        bs if use_cuda_graph else num_segments,
    )

    if base_output is None:
        output = torch.zeros((M, OUTPUT_DIM), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    _chunked_lora_expand_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        permutation=permutation,
        num_segs=num_segments,
        scalings=scalings,
        slice_offsets=slice_offsets,
        # constants
        NUM_SLICES=num_slices,
        OUTPUT_DIM=OUTPUT_DIM,
        MAX_RANK=MAX_RANK,
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
        slice_offsets,
        max_slice_size,
        base_output=None,
    ):
        return chunked_sgmv_lora_expand_forward(
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
            slice_offsets,
            max_slice_size,
            base_output,
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
        slice_offsets,
        max_slice_size,
        base_output=None,
    ):
        from sglang.srt.lora.triton_ops.chunked_sgmv_expand import (
            chunked_sgmv_lora_expand_forward,
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
        return chunked_sgmv_lora_expand_forward(
            x, weights, batch_info, slice_offsets, max_slice_size, base_output
        )


def get_inputs():
    batch_size = 4
    max_seq_len = 32
    num_slices = 3  # QKV
    max_rank = 64  # 最大 LoRA rank
    input_dim = 4096
    output_dim = 6144  # 4096 (Q) + 1024 (K) + 1024 (V)
    
    # 生成序列长度
    seq_lengths = [max_seq_len] * batch_size
    total_seq_len = sum(seq_lengths)
    
    # x: (total_seq_len, num_slices * max_rank) - 这是 shrink 操作的输出
    # 模拟 reference_shrink 的输出
    x = torch.randn(
        total_seq_len, num_slices * max_rank, dtype=torch.float16
    )
    
    num_loras = 2
    weights = torch.randn(
        num_loras, output_dim, max_rank, dtype=torch.float16
    )
    
    # slice_offsets: boundaries for different slices
    # [0, 4096, 5120, 6144] 表示 Q: [0, 4096), K: [4096, 5120), V: [5120, 6144)
    slice_offsets = torch.tensor([0, 4096, 5120, 6144], dtype=torch.int32)
    max_slice_size = 4096
    
    num_segments = batch_size
    seg_indptr = torch.zeros(num_segments + 1, dtype=torch.int32)
    for i in range(1, num_segments + 1):
        seg_indptr[i] = seg_indptr[i - 1] + seq_lengths[i - 1]
    
    weight_indices = torch.zeros(num_segments, dtype=torch.int32)  # 都使用第一个 LoRA
    lora_ranks = torch.tensor([max_rank, 32], dtype=torch.int32)  # 两个 LoRA 的 rank
    scalings = torch.ones(num_loras, dtype=torch.float32)  # 缩放因子
    
    # permutation: 逻辑索引到物理索引的映射（这里假设没有重排序）
    permutation = torch.arange(total_seq_len, dtype=torch.int32)
    
    use_cuda_graph = False
    bs = batch_size
    max_len = max_seq_len
    seg_lens = torch.tensor(seq_lengths, dtype=torch.int32)
    
    base_output = None
    
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
        slice_offsets,
        max_slice_size,
        base_output,
    ]


def get_init_inputs():
    return []

