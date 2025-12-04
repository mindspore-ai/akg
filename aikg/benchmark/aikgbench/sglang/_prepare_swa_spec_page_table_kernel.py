import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件：python/sglang/srt/layers/attention/flashattention_backend.py
# SGLang函数：_prepare_swa_spec_page_table_kernel
# 实现类型：Triton kernel
# 功能：准备SWA规范页面表
# 测试文件：无
# 输入参考：根据源文件中的函数签名和使用方式推断
# ============================================================================

# ============================================================================
# 以下是从SGLang直接复制的Triton Kernel实现
# ============================================================================

@triton.jit
def _prepare_swa_spec_page_table_kernel(
    dst_ptr,
    src_a_ptr,
    src_b_ptr,
    seq_len_a_ptr,
    seq_len_b_ptr,
    dst_stride_m,
    dst_stride_n,
    a_stride_m,
    a_stride_n,
    b_stride_m,
    b_stride_n,
    LEN_A: tl.constexpr,
    LEN_B: tl.constexpr,
    REPEAT_STEP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx_a = pid_m // REPEAT_STEP
    idx_b = pid_m
    seq_len_a = tl.load(seq_len_a_ptr + idx_a)
    seq_len_b = tl.load(seq_len_b_ptr + idx_b)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_len = seq_len_a + seq_len_b

    if pid_n * BLOCK_N >= total_len:
        return

    mask = offs_n < total_len
    dst = dst_ptr + pid_m * dst_stride_m + offs_n * dst_stride_n

    if (pid_n + 1) * BLOCK_N < seq_len_a:
        a_ptr = src_a_ptr + idx_a * a_stride_m + offs_n * a_stride_n
        a_mask = mask & (offs_n < LEN_A)
        val = tl.load(a_ptr, mask=a_mask, other=0)
        tl.store(dst, val, mask=mask)
    elif pid_n * BLOCK_N >= seq_len_a:
        offs_b = offs_n - seq_len_a
        b_ptr = src_b_ptr + idx_b * b_stride_m + offs_b * b_stride_n
        b_mask = mask & (offs_b < LEN_B)
        val = tl.load(b_ptr, mask=b_mask, other=0)
        tl.store(dst, val, mask=mask)
    else:
        # mixed part
        a_offs = offs_n
        a_mask = (a_offs < seq_len_a) & (a_offs < LEN_A)
        a_ptr = src_a_ptr + idx_a * a_stride_m + a_offs * a_stride_n
        a_val = tl.load(a_ptr, mask=a_mask, other=0)

        b_offs = offs_n - seq_len_a
        b_mask = (b_offs >= 0) & (b_offs < seq_len_b) & (b_offs < LEN_B)
        b_ptr = src_b_ptr + idx_b * b_stride_m + b_offs * b_stride_n
        b_val = tl.load(b_ptr, mask=b_mask, other=0)

        result = tl.where(offs_n < seq_len_a, a_val, b_val)
        tl.store(dst, result, mask=mask)


def prepare_swa_spec_page_table_triton(
    page_table_dst: torch.Tensor,
    page_table_a: torch.Tensor,
    page_table_b: torch.Tensor,  # expand page table
    seq_len_a: torch.Tensor,
    seq_len_b: torch.Tensor,  # expand seq lens
    speculative_num_draft_tokens: int,
):
    # concat page_table and expand page_table by kv seq length
    bs = seq_len_a.numel()
    bs_expand = seq_len_b.numel()
    assert bs_expand == bs * speculative_num_draft_tokens

    LEN_A = page_table_a.shape[1]
    LEN_B = page_table_b.shape[1]
    LEN_OUT = LEN_A + LEN_B
    REPEAT_STEP = speculative_num_draft_tokens
    BLOCK_N = 256

    grid = (bs_expand, triton.cdiv(LEN_OUT, BLOCK_N))
    _prepare_swa_spec_page_table_kernel[grid](
        page_table_dst,
        page_table_a,
        page_table_b,
        seq_len_a,
        seq_len_b,
        page_table_dst.stride(0),
        page_table_dst.stride(1),
        page_table_a.stride(0),
        page_table_a.stride(1),
        page_table_b.stride(0),
        page_table_b.stride(1),
        LEN_A=LEN_A,
        LEN_B=LEN_B,
        REPEAT_STEP=REPEAT_STEP,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """直接使用复制的Triton Kernel实现"""
    def __init__(self, speculative_num_draft_tokens: int = 1):
        super().__init__()
        self.speculative_num_draft_tokens = speculative_num_draft_tokens

    def forward(
        self,
        page_table_dst: torch.Tensor,
        page_table_a: torch.Tensor,
        page_table_b: torch.Tensor,
        seq_len_a: torch.Tensor,
        seq_len_b: torch.Tensor,
    ) -> torch.Tensor:
        prepare_swa_spec_page_table_triton(
            page_table_dst,
            page_table_a,
            page_table_b,
            seq_len_a,
            seq_len_b,
            self.speculative_num_draft_tokens
        )
        return page_table_dst


class ModelSGLang(nn.Module):
    """SGLang实现的PyTorch版本"""
    def __init__(self, speculative_num_draft_tokens: int = 1):
        super().__init__()
        self.speculative_num_draft_tokens = speculative_num_draft_tokens

    def forward(
        self,
        page_table_dst: torch.Tensor,
        page_table_a: torch.Tensor,
        page_table_b: torch.Tensor,
        seq_len_a: torch.Tensor,
        seq_len_b: torch.Tensor,
    ) -> torch.Tensor:
        # PyTorch实现版本
        # 这是一个简化的PyTorch实现，用于演示目的
        bs = seq_len_a.numel()
        bs_expand = seq_len_b.numel()
        assert bs_expand == bs * self.speculative_num_draft_tokens
        
        LEN_A = page_table_a.shape[1]
        LEN_B = page_table_b.shape[1]
        
        # 简化实现：直接拼接page_table_a和page_table_b
        for i in range(bs_expand):
            idx_a = i // self.speculative_num_draft_tokens
            seq_a_len = seq_len_a[idx_a].item()
            seq_b_len = seq_len_b[i].item()
            
            # 复制page_table_a的内容
            page_table_dst[i, :seq_a_len] = page_table_a[idx_a, :seq_a_len]
            # 复制page_table_b的内容
            page_table_dst[i, seq_a_len:seq_a_len+seq_b_len] = page_table_b[i, :seq_b_len]
            
        return page_table_dst


def get_inputs():
    """生成测试输入"""
    dtype = torch.int32
    
    # 创建测试张量
    bs = 2
    speculative_num_draft_tokens = 3
    bs_expand = bs * speculative_num_draft_tokens
    
    LEN_A = 10
    LEN_B = 5
    LEN_OUT = LEN_A + LEN_B
    
    page_table_dst = torch.zeros((bs_expand, LEN_OUT), dtype=dtype)
    page_table_a = torch.randint(0, 100, (bs, LEN_A), dtype=dtype)
    page_table_b = torch.randint(0, 100, (bs_expand, LEN_B), dtype=dtype)
    seq_len_a = torch.tensor([8, 6], dtype=torch.int32)
    seq_len_b = torch.tensor([3, 2, 4, 3, 2, 4], dtype=torch.int32)  # bs_expand长度
    
    return [page_table_dst, page_table_a, page_table_b, seq_len_a, seq_len_b]


def get_init_inputs():
    """生成初始化参数"""
    return [3]  # speculative_num_draft_tokens