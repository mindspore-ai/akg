import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件：python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
# SGLang函数：_log_softmax_kernel
# 实现类型：Triton kernel
# 功能：使用Triton实现的log_softmax计算
# 测试文件：test/nightly/test_batch_invariant_ops.py
# 输入参考：根据源文件中的函数签名和test_batch_invariant_ops.py中的测试用例推断
# ============================================================================


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    # Load first block to infer dtype and initialize max_val with correct type
    col_idx_init = tl.arange(0, BLOCK_SIZE)
    mask_init = col_idx_init < n_cols
    vals_init = tl.load(
        row_start_ptr + col_idx_init, mask=mask_init, other=-float("inf")
    )
    max_val = tl.max(vals_init)

    # Continue with remaining blocks
    for col_offset in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    # Initialize sum_exp with correct dtype by using tl.sum on a zero vector
    sum_exp = tl.sum(tl.zeros([1], dtype=max_val.dtype))

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


# ============================================================================
# AIKGBench标准接口
# ============================================================================
def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """使用Triton kernel计算log_softmax"""
        return log_softmax(input_tensor)


class ModelSGLang(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """使用PyTorch原生函数计算log_softmax（作为对比基准）"""
        return torch.nn.functional.log_softmax(input_tensor, dim=-1)


# ============================================================================
# 测试输入生成函数
# ============================================================================

def get_inputs():
    """生成测试输入"""
    batch_size = 8
    feature_dim = 64
    input_tensor = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    return [input_tensor]

def get_init_inputs():
    """生成模型初始化所需的输入"""
    return []