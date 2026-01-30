# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/lora/triton_ops/sgemm_lora_b.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   sgemm_lora_b_fwd(x, weights, batch_info, base_output)
# Triton Kernel:
#   sgemm_lora_b_kernel - 这个内核函数主要实现计算LoRA B矩阵的乘积。
# ============================================================================

from typing import Optional

import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def _sgemm_lora_b_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Matrix dimensions
    N,  # output_dim
    K,  # r
    # Strides
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    # Information on sequence lengths and weight id
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    # Meta parameters
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # For fused output scaling
    scalings,
):
    """
    Computes a segmented batched matrix multiplication for the LoRA B matrix
    and adds the result to the output in-place.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (torch.Tensor): The intermediate tensor from the LoRA 'A' multiplication,
            of shape `(s, K)`, where `s` is the total number of tokens.
        weights (torch.Tensor): The LoRA 'B' weights for all available adapters,
            with shape `(num_lora, N, K)`.
        output (torch.Tensor): The output tensor of shape `(s, N)`. This can be
            the base model's output for a fused add operation.
    """

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter
    K = tl.minimum(K, rank)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = (x + seg_start * x_stride_0) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (output + seg_start * output_stride_0) + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = s_offset[:, None] < seg_len
    partial_sum += tl.load(output_ptr, mask=output_mask)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_b_fwd(
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
    base_output: torch.Tensor = None,
) -> torch.Tensor:
    # x: (s, max_r)
    # weights: (num_lora, output_dim, max_r)
    # output: (s, output_dim)
    # output_dim is much larger than max_r

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    S = x.shape[0]
    N = weights.shape[-2]
    R = weights.shape[-1]
    assert x.shape[-1] == R

    # Block shapes
    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_N = 256

    grid = (
        triton.cdiv(max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
        bs,
    )

    if base_output is None:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
        N,
        R,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        seg_lens,
        seg_indptr,
        weight_indices,
        lora_ranks,
        BLOCK_S,
        BLOCK_N,
        BLOCK_R,
        scalings,
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
        base_output=None,
    ):
        return sgemm_lora_b_fwd(
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
        base_output=None,
    ):
        from sglang.srt.lora.triton_ops.sgemm_lora_b import sgemm_lora_b_fwd
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
        return sgemm_lora_b_fwd(x, weights, batch_info, base_output)


def get_inputs():
    batch_size = 4
    max_seq_len = 32
    max_rank = 64  # 最大 LoRA rank (r)
    output_dim = 4096  # 输出维度
    
    # 生成序列长度
    seq_lengths = [max_seq_len] * batch_size
    total_seq_len = sum(seq_lengths)
    
    # x: (total_seq_len, max_r) - LoRA A 的输出（中间激活）
    # 用户说可以构造一个 tensor 就行
    x = torch.randn(
        total_seq_len, max_rank, dtype=torch.float16
    )
    
    # weights: (num_lora, output_dim, max_r) - LoRA B weights
    # 假设有 2 个 LoRA adapters
    num_loras = 2
    weights = torch.randn(
        num_loras, output_dim, max_rank, dtype=torch.float16
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
        base_output,
    ]


def get_init_inputs():
    return []

