import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件：sglang/python/sglang/srt/layers/attention/fla/cumsum.py
# SGLang函数：chunk_local_cumsum_scalar_kernel
# 实现类型：Triton kernel
# 功能：分块计算局部累积和
# 测试文件：无
# 输入参考：根据源文件中的函数签名和chunk.py中的使用方式推断
# ============================================================================

# ============================================================================
# 以下是从SGLang直接复制的Triton Kernel实现
# ============================================================================

# Constants from the original implementation
def check_shared_mem():
    # Simplified implementation
    return True

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]

def prepare_chunk_indices(cu_seqlens, chunk_size):
    # Simplified implementation
    batch_size = len(cu_seqlens) - 1
    chunk_indices = []
    for b in range(batch_size):
        start, end = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
        for i in range(start, end, chunk_size):
            chunk_indices.append([b, i])
    return torch.tensor(chunk_indices, dtype=torch.int32, device=cu_seqlens.device)

@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))

def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=8,
        num_stages=3,
    )
    return g

# ============================================================================
# AIKGBench标准接口
# ============================================================================
class Model(nn.Module):
    """直接使用复制的Triton Kernel实现"""
    def __init__(self, chunk_size: int = 64, reverse: bool = False, 
                 scale: float = None, head_first: bool = False, 
                 output_dtype: Optional[torch.dtype] = torch.float):
        super().__init__()
        self.chunk_size = chunk_size
        self.reverse = reverse
        self.scale = scale
        self.head_first = head_first
        self.output_dtype = output_dtype

    def forward(self, g: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=self.chunk_size,
            reverse=self.reverse,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            head_first=self.head_first,
            output_dtype=self.output_dtype
        )

class ModelSGLang(nn.Module):
    """基于源文件的PyTorch实现"""
    def __init__(self, chunk_size: int = 64, reverse: bool = False, 
                 scale: float = None, head_first: bool = False, 
                 output_dtype: Optional[torch.dtype] = torch.float):
        super().__init__()
        self.chunk_size = chunk_size
        self.reverse = reverse
        self.scale = scale
        self.head_first = head_first
        self.output_dtype = output_dtype

    def forward(self, g: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Simplified PyTorch implementation based on the original function
        if self.head_first:
            B, H, T = g.shape
            # Transpose to make it easier to process
            g = g.transpose(1, 2)  # [B, T, H]
        else:
            B, T, H = g.shape
        
        # Create output tensor
        output = torch.empty_like(g, dtype=self.output_dtype or g.dtype)
        
        # Process each batch and head
        for b in range(B):
            for h in range(H):
                if cu_seqlens is not None:
                    # Variable length sequences
                    start, end = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
                    seq_len = end - start
                    g_seq = g[b, start:end, h]
                else:
                    # Fixed length sequences
                    g_seq = g[b, :, h]
                    seq_len = T
                
                # Process in chunks
                chunked_output = []
                for i in range(0, seq_len, self.chunk_size):
                    chunk_end = min(i + self.chunk_size, seq_len)
                    chunk = g_seq[i:chunk_end]
                    
                    # Compute cumsum
                    if self.reverse:
                        # Reverse cumsum
                        chunk_sum = torch.sum(chunk)
                        chunk_cumsum = torch.cumsum(chunk.flip(dims=[0]), dim=0).flip(dims=[0])
                    else:
                        # Forward cumsum
                        chunk_cumsum = torch.cumsum(chunk, dim=0)
                    
                    # Apply scale if provided
                    if self.scale is not None:
                        chunk_cumsum *= self.scale
                    
                    chunked_output.append(chunk_cumsum)
                
                # Concatenate chunks
                full_output = torch.cat(chunked_output)
                
                # Store result
                if cu_seqlens is not None:
                    output[b, start:end, h] = full_output
                else:
                    output[b, :, h] = full_output
        
        # Transpose back if needed
        if self.head_first:
            output = output.transpose(1, 2)  # [B, H, T]
        
        return output

def get_inputs():
    """生成测试输入"""
    # Use fixed parameters based on usage in chunk.py
    batch_size = 1
    seq_len = 10
    num_heads = 4
    
    # Create random input tensor (B, T, H) format as used in chunk.py
    g = torch.randn(batch_size, seq_len, num_heads)
    
    # For variable length sequences
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len)
    
    return [g, cu_seqlens]

def get_init_inputs():
    """生成初始化参数"""
    return [64, False, None, False, torch.float]  # chunk_size, reverse, scale, head_first, output_dtype