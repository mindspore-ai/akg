import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: sgl-kernel/tests/test_moe_align.py
# 测试文件: sgl-kernel/tests/test_moe_align.py
# SGLang API调用:
#   moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids,
#                         expert_ids, num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids)
# Triton Kernel:
#   moe_align_block_size_stage1-4 - Aligns token distribution across experts
#                                    to be compatible with block size for matmul
# 标杆实现:
#   sgl_kernel.moe_align_block_size - CUDA/C++ implementation
# ============================================================================


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, topk_ids, num_experts, block_size):
        """
        Args:
            topk_ids: [total_tokens, top_k] expert indices for each token
            num_experts: total number of experts
            block_size: block size for matrix multiplication
        Returns:
            sorted_token_ids: sorted token indices by expert
            expert_ids: expert index for each block
            num_tokens_post_pad: total tokens after padding
        """
        numel = topk_ids.numel()
        grid = (num_experts,)

        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
        sorted_token_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
        )
        sorted_token_ids.fill_(topk_ids.numel())

        max_num_m_blocks = max_num_tokens_padded // block_size
        expert_ids = torch.zeros(
            (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
        )
        num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

        tokens_cnts = torch.zeros(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        )
        cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
        tokens_per_thread = ceil_div(numel, num_experts)

        moe_align_block_size_stage1[grid](
            topk_ids,
            tokens_cnts,
            num_experts,
            numel,
            tokens_per_thread,
        )
        moe_align_block_size_stage2[grid](
            tokens_cnts,
            num_experts,
        )
        moe_align_block_size_stage3[(1,)](
            num_tokens_post_pad,
            tokens_cnts,
            cumsum,
            num_experts,
            block_size,
        )
        moe_align_block_size_stage4[grid](
            topk_ids,
            sorted_token_ids,
            expert_ids,
            tokens_cnts,
            cumsum,
            num_experts,
            block_size,
            numel,
            tokens_per_thread,
        )

        return sorted_token_ids, expert_ids, num_tokens_post_pad


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, topk_ids, num_experts, block_size):
        """
        Reference implementation using sgl_kernel CUDA implementation
        """
        from sgl_kernel import moe_align_block_size

        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
        sorted_token_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
        )

        max_num_m_blocks = max_num_tokens_padded // block_size
        expert_ids = torch.zeros(
            (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
        )
        num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
        cumsum_buffer = torch.empty(
            (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
        )

        moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
        )

        return sorted_token_ids, expert_ids, num_tokens_post_pad


def get_inputs():
    # Example dimensions for MOE align block size
    num_tokens = 512
    topk = 8
    num_experts = 64
    block_size = 64

    # Generate random expert assignments
    topk_ids = torch.argsort(
        torch.rand(num_tokens, num_experts + 1),
        dim=1
    )[:, :topk].contiguous().flatten().to(torch.int32)

    return [topk_ids, num_experts + 1, block_size]


def get_init_inputs():
    return []
