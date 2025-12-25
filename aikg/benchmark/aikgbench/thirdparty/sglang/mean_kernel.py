import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件：python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
# SGLang函数：mean_kernel
# 实现类型：Triton kernel
# 功能：计算张量沿单个维度的平均值
# 测试文件：test/nightly/test_batch_invariant_ops.py
# 输入参考：根据源文件中的函数签名和test_batch_invariant_ops.py中的测试用例推断
# ============================================================================

# ============================================================================
# 以下是从SGLang直接复制的Triton Kernel实现
# ============================================================================

@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert (
        -input.ndim <= dim < input.ndim
    ), f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output

# ============================================================================
# AIKGBench标准接口
# ============================================================================
class Model(nn.Module):
    """直接使用复制的Triton Kernel实现"""
    def __init__(self, dim: int = 0, keepdim: bool = False, dtype: torch.dtype | None = None):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 为了兼容性，将输入移到CUDA上
        if not input.is_cuda:
            input = input.cuda()
        return mean_dim(input, dim=self.dim, keepdim=self.keepdim, dtype=self.dtype)


class ModelSGLang(nn.Module):
    """sglang实现"""
    def __init__(self, dim: int = 0, keepdim: bool = False, dtype: torch.dtype | None = None):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 基于batch_invariant_ops.py中的mean_batch_invariant函数实现
        assert self.dtype is None or self.dtype == torch.float32, f"unsupported dtype: {self.dtype}"
        
        # 处理dim参数，确保是列表格式
        if isinstance(self.dim, int):
            dims = [self.dim]
        elif isinstance(self.dim, (list, tuple)):
            dims = list(self.dim)
        else:
            dims = self.dim
            
        if len(dims) == 1:
            return mean_dim(input, dims[0], keepdim=self.keepdim)
        else:
            assert input.dtype in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
            }, "only float types supported for now"
            n_elems = 1
            for d in dims:
                n_elems *= input.shape[d]
            return torch.sum(input, dim=dims, keepdim=self.keepdim, dtype=torch.float32) / n_elems


def get_inputs():
    """生成测试输入"""
    shape = (16, 32, 64)
    dtype = torch.float32
    
    # 创建随机输入张量，不指定device，让Model的forward自动处理
    input = torch.randn(*shape, dtype=dtype)
    
    return [input]


def get_init_inputs():
    """生成初始化参数"""
    # 默认参数：dim=1, keepdim=False, dtype=None
    return [1, False, None]