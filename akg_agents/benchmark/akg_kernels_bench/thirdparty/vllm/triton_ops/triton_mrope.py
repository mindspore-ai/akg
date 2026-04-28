# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/rotary_embedding/mrope.py
# vLLM函数: triton_mrope
# 实现类型: Triton kernel
# 测试文件: 无专门测试文件
# 输入参考: q=[num_tokens, num_heads*head_size], k=[num_tokens, num_kv_heads*head_size]
# ============================================================================

import torch
import torch.nn as nn

# ============================================================================
# 从vLLM复制的Triton kernel实现
# ============================================================================
import triton
import triton.language as tl


@triton.jit
def _triton_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)


def triton_mrope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list,
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
):
    """从vLLM复制的host侧调用代码"""
    num_tokens = q.shape[0]
    num_heads = q.shape[1] // head_size
    num_kv_heads = k.shape[1] // head_size

    pad_n_qh = triton.next_power_of_2(num_heads)
    pad_n_kh = triton.next_power_of_2(num_kv_heads)
    pad_hd = triton.next_power_of_2(head_size)

    _triton_mrope_forward[(num_tokens,)](
        q,
        k,
        cos,
        sin,
        num_tokens,
        num_heads,
        num_kv_heads,
        head_size,
        rotary_dim,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
    )
    return q, k


# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """原生实现（直接调用复制的Triton kernel）"""
    
    def __init__(self, head_size: int = 128, rotary_dim: int = 128):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.mrope_section = [32, 32, 32]  # T, H, W sections
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple:
        return triton_mrope_impl(
            q.clone(), k.clone(), cos, sin, 
            self.mrope_section, self.head_size, self.rotary_dim, False
        )


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""
    
    def __init__(self, head_size: int = 128, rotary_dim: int = 128):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.mrope_section = [32, 32, 32]
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple:
        from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
        return triton_mrope(
            q.clone(), k.clone(), cos, sin,
            self.mrope_section, self.head_size, self.rotary_dim, False
        )


def get_inputs():
    """生成测试输入"""
    num_tokens = 128
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    rotary_dim = 128
    
    q = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    k = torch.randn(num_tokens, num_kv_heads * head_size, dtype=torch.float16)
    cos = torch.randn(3, num_tokens, head_size // 2, dtype=torch.float32)
    sin = torch.randn(3, num_tokens, head_size // 2, dtype=torch.float32)
    
    return [q, k, cos, sin]


def get_init_inputs():
    """获取初始化参数"""
    return [128, 128]  # head_size, rotary_dim

