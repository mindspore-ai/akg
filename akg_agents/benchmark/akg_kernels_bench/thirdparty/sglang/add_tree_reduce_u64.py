# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/multimodal.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   add_tree_reduce_u64(input_tensor, CHUNK=1024)
# Triton Kernel:
#   add_tree_reduce_u64_kernel - Tree reduction for uint64 tensor
# ============================================================================
import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def add_tree_reduce_u64_kernel(in_ptr, out_ptr, n_elems, CHUNK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * CHUNK
    h = tl.zeros((), dtype=tl.uint64)
    for i in tl.static_range(0, CHUNK):
        idx = start + i
        m = idx < n_elems
        v = tl.load(in_ptr + idx, mask=m, other=0).to(tl.uint64)
        h += v
    tl.store(out_ptr + pid, h)


def add_tree_reduce_u64(
    input_tensor: torch.Tensor,
    CHUNK: int = 1024,
):
    """
    Tree reduction for uint64 tensor.
    
    Args:
        input_tensor: Input tensor of int64 values (representing uint64)
        CHUNK: Chunk size for reduction (default: 1024)
    
    Returns:
        Output tensor with reduced values
    """
    assert input_tensor.dtype == torch.int64, "Input must be int64 (representing uint64)"
    n_elems = input_tensor.numel()
    
    if n_elems == 0:
        return torch.empty((0,), dtype=torch.int64, device=input_tensor.device)
    
    grid = (triton.cdiv(n_elems, CHUNK),)
    output = torch.empty(grid[0], dtype=torch.int64, device=input_tensor.device)
    
    add_tree_reduce_u64_kernel[grid](
        input_tensor,
        output,
        n_elems,
        CHUNK=CHUNK,
    )
    
    return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input_tensor, CHUNK=1024):
        return add_tree_reduce_u64(input_tensor, CHUNK)


class ModelSglang(nn.Module):
    def __init__(self):
        super(ModelSglang, self).__init__()
    
    def forward(self, input_tensor, CHUNK=1024):
        from sglang.srt.layers.multimodal import add_tree_reduce_u64_kernel
        
        n_elems = input_tensor.numel()
        
        if n_elems == 0:
            return torch.empty((0,), dtype=torch.int64, device=input_tensor.device)
        
        grid = (triton.cdiv(n_elems, CHUNK),)
        output = torch.empty(grid[0], dtype=torch.int64, device=input_tensor.device)
        
        add_tree_reduce_u64_kernel[grid](
            input_tensor,
            output,
            n_elems,
            CHUNK=CHUNK,
        )
        
        return output


def get_inputs():
    n_elems = 2048  # 输入元素数量（模拟 partials 的数量）
    CHUNK = 1024  # 归约的 chunk 大小（默认值）
    
    input_tensor = torch.randint(0, 2**63 -2, (n_elems,), dtype=torch.int64)
    
    return [input_tensor, CHUNK]


def get_init_inputs():
    return []

