# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/multimodal.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   hash_tiles32_blocked(input_tensor, seed=0x243F6A88, tile_words=8192, block_words=256, use_cg=True, num_warps=4, num_stages=4)
# Triton Kernel:
#   hash_tiles32_kernel_blocked - Hash tiles using blocked kernel
# ============================================================================


import torch
import triton
import triton.language as tl
import torch.nn as nn

FMIX32_C1 = 0x85EBCA6B
FMIX32_C2 = 0xC2B2AE35
POS_C1 = 0x27D4EB2D
POS_C2 = 0x165667B1

@triton.jit
def _rotl32(x, r: tl.constexpr):
    return (x << r) | (x >> (32 - r))

@triton.jit
def _fmix32(x, C1: tl.constexpr, C2: tl.constexpr):
    c1 = tl.full((), C1, tl.uint32)
    c2 = tl.full((), C2, tl.uint32)
    x ^= x >> 16
    x = x * c1
    x ^= x >> 13
    x = x * c2
    x ^= x >> 16
    return x


@triton.jit
def hash_tiles32_kernel_blocked(
    in_ptr,
    out_ptr,
    n_u32,
    seed1,
    seed2,
    FM_C1: tl.constexpr,
    FM_C2: tl.constexpr,
    POS_A: tl.constexpr,
    POS_B: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_CG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base = pid * TILE

    s1 = tl.full((), seed1, tl.uint32)
    s2 = tl.full((), seed2, tl.uint32)
    posA = tl.full((), POS_A, tl.uint32)
    posB = tl.full((), POS_B, tl.uint32)

    h1 = tl.zeros((), dtype=tl.uint32)
    h2 = tl.zeros((), dtype=tl.uint32)

    for off in tl.static_range(0, TILE, BLOCK):
        idx = base + off + tl.arange(0, BLOCK)
        m = idx < n_u32

        if USE_CG:
            v = tl.load(in_ptr + idx, mask=m, other=0, cache_modifier=".cg")
        else:
            v = tl.load(in_ptr + idx, mask=m, other=0)
        v = v.to(tl.uint32)

        iu = idx.to(tl.uint32)
        p1 = (iu * posA + s1) ^ _rotl32(iu, 15)
        p2 = (iu * posB + s2) ^ _rotl32(iu, 13)

        k1 = _fmix32(v ^ p1, C1=FM_C1, C2=FM_C2)
        k2 = _fmix32(v ^ p2, C1=FM_C1, C2=FM_C2)

        zero32 = tl.zeros_like(k1)
        k1 = tl.where(m, k1, zero32)
        k2 = tl.where(m, k2, zero32)

        h1 += tl.sum(k1, axis=0).to(tl.uint32)
        h2 += tl.sum(k2, axis=0).to(tl.uint32)

    nbytes = tl.full((), n_u32 * 4, tl.uint32)
    h1 ^= nbytes
    h2 ^= nbytes
    h1 = _fmix32(h1, C1=FM_C1, C2=FM_C2)
    h2 = (
        _fmix32(h2, C1=FMIX32_C1, C2=FMIX32_C2)
        if False
        else _fmix32(h2, C1=FM_C1, C2=FM_C2)
    )

    out = (h1.to(tl.uint64) << 32) | h2.to(tl.uint64)
    tl.store(out_ptr + pid, out)


def _as_uint32_words(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor to uint32 view."""
    tb = t.contiguous().view(torch.uint8)
    nbytes = tb.numel()
    pad = (4 - (nbytes & 3)) & 3
    if pad:
        tb_p = torch.empty(nbytes + pad, dtype=torch.uint8, device=tb.device)
        tb_p[:nbytes].copy_(tb)
        tb_p[nbytes:].zero_()
        tb = tb_p
    return tb.view(torch.uint32)


def hash_tiles32_blocked(
    input_tensor: torch.Tensor,
    seed: int = 0x243F6A88,
    tile_words: int = 8192,
    block_words: int = 256,
    use_cg: bool = True,
    num_warps: int = 4,
    num_stages: int = 4,
):
    """
    Hash tiles using blocked kernel.
    
    Args:
        input_tensor: Input tensor (any dtype, will be converted to uint32 view)
        seed: Hash seed (default: 0x243F6A88)
        tile_words: Number of uint32 words per tile (default: 8192)
        block_words: Number of uint32 words per block (default: 256)
        use_cg: Use cache global modifier (default: True)
        num_warps: Number of warps (default: 4)
        num_stages: Number of stages (default: 4)
    
    Returns:
        Output tensor of uint64 partial hash values
    """
    u32 = _as_uint32_words(input_tensor)
    n = u32.numel()
    
    if n == 0:
        return torch.empty((0,), dtype=torch.int64, device=u32.device)
    
    grid = (triton.cdiv(n, tile_words),)
    partials = torch.empty(grid[0], dtype=torch.int64, device=u32.device)
    
    hash_tiles32_kernel_blocked[grid](
        u32,
        partials,
        n,
        seed1=seed & 0xFFFFFFFF,
        seed2=((seed * 0x9E3779B1) ^ 0xDEADBEEF) & 0xFFFFFFFF,
        FM_C1=FMIX32_C1,
        FM_C2=FMIX32_C2,
        POS_A=POS_C1,
        POS_B=POS_C2,
        TILE=tile_words,
        BLOCK=block_words,
        USE_CG=use_cg,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return partials


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(
        self,
        input_tensor,
        seed=0x243F6A88,
        tile_words=8192,
        block_words=256,
        use_cg=True,
        num_warps=4,
        num_stages=4,
    ):
        return hash_tiles32_blocked(
            input_tensor,
            seed=seed,
            tile_words=tile_words,
            block_words=block_words,
            use_cg=use_cg,
            num_warps=num_warps,
            num_stages=num_stages,
        )


class ModelSglang(nn.Module):
    def __init__(self):
        super(ModelSglang, self).__init__()
    
    def forward(
        self,
        input_tensor,
        seed=0x243F6A88,
        tile_words=8192,
        block_words=256,
        use_cg=True,
        num_warps=4,
        num_stages=4,
    ):
        from sglang.srt.layers.multimodal import hash_tiles32_kernel_blocked
        from sglang.srt.layers.multimodal import _as_uint32_words
        
        u32 = _as_uint32_words(input_tensor)
        n = u32.numel()
        
        if n == 0:
            return torch.empty((0,), dtype=torch.int64, device=u32.device)
        
        grid = (triton.cdiv(n, tile_words),)
        partials = torch.empty(grid[0], dtype=torch.int64, device=u32.device)
        
        hash_tiles32_kernel_blocked[grid](
            u32,
            partials,
            n,
            seed1=seed & 0xFFFFFFFF,
            seed2=((seed * 0x9E3779B1) ^ 0xDEADBEEF) & 0xFFFFFFFF,
            FM_C1=FMIX32_C1,
            FM_C2=FMIX32_C2,
            POS_A=POS_C1,
            POS_B=POS_C2,
            TILE=tile_words,
            BLOCK=block_words,
            USE_CG=use_cg,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        
        return partials


def get_inputs():
    n_elements = 16384 
    input_tensor = torch.randn(n_elements, dtype=torch.float32)
    
    seed = 0x243F6A88
    tile_words = 8192
    block_words = 256
    use_cg = True
    num_warps = 4
    num_stages = 4
    
    return [
        input_tensor,
        seed,
        tile_words,
        block_words,
        use_cg,
        num_warps,
        num_stages,
    ]


def get_init_inputs():
    return []

