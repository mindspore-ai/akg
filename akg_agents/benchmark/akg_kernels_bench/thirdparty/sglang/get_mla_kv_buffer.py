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
#   get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
# Triton Kernel:
#   get_mla_kv_buffer_kernel - 从MLA KV缓冲区获取nope和rope部分
# ============================================================================


@triton.jit
def get_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    loc = tl.load(loc_ptr + pid_loc)
    loc_src_ptr = kv_buffer_ptr + loc * buffer_stride

    nope_offs = tl.arange(0, nope_dim)
    nope_src_ptr = loc_src_ptr + nope_offs
    nope_src = tl.load(nope_src_ptr)

    tl.store(
        cache_k_nope_ptr + pid_loc * nope_stride + nope_offs,
        nope_src,
    )

    rope_offs = tl.arange(0, rope_dim)
    rope_src_ptr = loc_src_ptr + nope_dim + rope_offs
    rope_src = tl.load(rope_src_ptr)
    tl.store(
        cache_k_rope_ptr + pid_loc * rope_stride + rope_offs,
        rope_src,
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, kv_buffer, loc, cache_k_nope, cache_k_rope):
        nope_dim = cache_k_nope.shape[-1]
        rope_dim = cache_k_rope.shape[-1]
        n_loc = loc.numel()
        grid = (n_loc,)

        get_mla_kv_buffer_kernel[grid](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
        )

        return cache_k_nope, cache_k_rope


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, kv_buffer, loc, cache_k_nope, cache_k_rope):
        from sglang.srt.mem_cache.utils import get_mla_kv_buffer_triton
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope


def get_inputs():
    # Example dimensions for MLA (Multi-head Latent Attention)
    n_loc = 16
    buffer_size = 1024
    nope_dim = 512
    rope_dim = 64
    total_dim = nope_dim + rope_dim
    dtype = torch.float16

    kv_buffer = torch.randn(buffer_size, total_dim, dtype=dtype)
    loc = torch.randint(0, buffer_size, (n_loc,), dtype=torch.int64)
    cache_k_nope = torch.empty(n_loc, nope_dim, dtype=dtype)
    cache_k_rope = torch.empty(n_loc, rope_dim, dtype=dtype)

    return [kv_buffer, loc, cache_k_nope, cache_k_rope]


def get_init_inputs():
    return []
