import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/mem_cache/utils.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   set_mla_kv_scale_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
# Triton Kernel:
#   set_mla_kv_scale_buffer_kernel - 将nope和rope缩放后写入MLA KV缓冲区
# ============================================================================


@triton.jit
def set_mla_kv_scale_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim  # Make sure don't cross the boundary

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    # Check each offs should read 'nope' or 'rope'
    is_nope = offs < nope_dim
    src_nope = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs, mask=mask & is_nope, other=0.0
    )
    src_rope = tl.load(
        cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
        mask=mask & ~is_nope,
        other=0.0,
    )

    # Combine nope + rope
    src = src_nope + src_rope
    tl.store(dst_ptr, src, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, kv_buffer, loc, cache_k_nope, cache_k_rope):
        nope_dim = cache_k_nope.shape[-1]
        rope_dim = cache_k_rope.shape[-1]
        total_dim = nope_dim + rope_dim
        BLOCK = 128
        n_loc = loc.numel()
        grid = (n_loc, triton.cdiv(total_dim, BLOCK))

        set_mla_kv_scale_buffer_kernel[grid](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK=BLOCK,
        )

        return kv_buffer


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, kv_buffer, loc, cache_k_nope, cache_k_rope):
        from sglang.srt.mem_cache.utils import set_mla_kv_scale_buffer_triton
        set_mla_kv_scale_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return kv_buffer


def get_inputs():
    # Example dimensions for MLA (Multi-head Latent Attention) with scale
    n_loc = 16
    buffer_size = 1024
    nope_dim = 512
    rope_dim = 64
    total_dim = nope_dim + rope_dim
    dtype = torch.float16

    kv_buffer = torch.randn(buffer_size, total_dim, dtype=dtype, device='cuda')
    loc = torch.randint(0, buffer_size, (n_loc,), dtype=torch.int64, device='cuda')
    cache_k_nope = torch.randn(n_loc, nope_dim, dtype=dtype, device='cuda')
    cache_k_rope = torch.randn(n_loc, rope_dim, dtype=dtype, device='cuda')

    return [kv_buffer, loc, cache_k_nope, cache_k_rope]


def get_init_inputs():
    return []
