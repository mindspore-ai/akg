# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/multimodal.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   rotl32(input_tensor, r=15)
# Triton Kernel:
#   rotl32_kernel - 这个内核函数主要实现将int32类型的值左移r位。
# ============================================================================
import torch
import triton
import triton.language as tl
import torch.nn as nn

from sglang.srt.layers.multimodal import _rotl32 as _rotl32_sgl


@triton.jit
def _rotl32(x, r: tl.constexpr):
    return (x << r) | (x >> (32 - r))


@triton.jit
def rotl32_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    r: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.int32)
    result = _rotl32(x, r=r)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def rotl32_kernel_sgl(
    input_ptr,
    output_ptr,
    n_elements,
    r: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.int32)
    result = _rotl32_sgl(x, r=r)
    tl.store(output_ptr + offsets, result, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input_tensor, r):
        assert input_tensor.dtype == torch.int32, "Input must be int32"
        n_elements = input_tensor.numel()
        
        output = torch.empty_like(input_tensor)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        rotl32_kernel[grid](
            input_tensor,
            output,
            n_elements,
            r=r,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output


class ModelSglang(nn.Module):
    def __init__(self):
        super(ModelSglang, self).__init__()
    
    def forward(self, input_tensor, r):
        assert input_tensor.dtype == torch.int32, "Input must be int32"
        n_elements = input_tensor.numel()
        
        output = torch.empty_like(input_tensor)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        rotl32_kernel_sgl[grid](
            input_tensor,
            output,
            n_elements,
            r=r,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output


def get_inputs():
    n_elements = 1024  # 元素数量
    
    input_tensor = torch.arange(n_elements, dtype=torch.int32)
    
    r = 15
    
    return [input_tensor, r]


def get_init_inputs():
    return []