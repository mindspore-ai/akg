# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/multimodal.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   fmix32(input_tensor, C1=0x85EBCA6B, C2=0xC2B2AE35)
# Triton Kernel:
#   fmix32_kernel - 这个内核函数主要实现MurmurHash3 的最终混合（finalization）步骤，用于将哈希值混合得更均匀，减少碰撞。
# ============================================================================
import torch
import triton
import triton.language as tl
import torch.nn as nn

FMIX32_C1 = 0x85EBCA6B
FMIX32_C2 = 0xC2B2AE35

from sglang.srt.layers.multimodal import _fmix32 as _fmix32_sgl


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
def fmix32_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.int32)
    result = _fmix32(x, C1=C1, C2=C2)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def fmix32_kernel_sgl(
    input_ptr,
    output_ptr,
    n_elements,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.int32)
    result = _fmix32_sgl(x, C1=C1, C2=C2)
    tl.store(output_ptr + offsets, result, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input_tensor, C1=None, C2=None):
        if C1 is None:
            C1 = FMIX32_C1
        if C2 is None:
            C2 = FMIX32_C2
        
        assert input_tensor.dtype == torch.int32, "Input must be int32 (representing uint32)"
        n_elements = input_tensor.numel()
        
        output = torch.empty_like(input_tensor)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fmix32_kernel[grid](
            input_tensor,
            output,
            n_elements,
            C1=C1,
            C2=C2,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output


class ModelSglang(nn.Module):
    def __init__(self):
        super(ModelSglang, self).__init__()
    
    def forward(self, input_tensor, C1=None, C2=None):
        if C1 is None:
            C1 = FMIX32_C1
        if C2 is None:
            C2 = FMIX32_C2
        
        assert input_tensor.dtype == torch.int32, "Input must be int32 (representing uint32)"
        n_elements = input_tensor.numel()
        
        output = torch.empty_like(input_tensor)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fmix32_kernel_sgl[grid](
            input_tensor,
            output,
            n_elements,
            C1=C1,
            C2=C2,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output


def get_inputs():
    n_elements = 1024  # 元素数量
    
    uint32_max = 2**32 - 1
    rand_int64 = torch.randint(0, uint32_max + 1, (n_elements,), dtype=torch.int64)
    input_tensor = rand_int64.to(torch.int32)

    C1 = FMIX32_C1
    C2 = FMIX32_C2
    
    return [input_tensor, C1, C2]


def get_init_inputs():
    return []