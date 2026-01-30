import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   moe_sum_reduce_triton(input, output, routed_scaling_factor)
# Triton Kernel:
#   _moe_sum_reduce_kernel - MOE sum reduce operation for combining expert outputs
# 标杆实现:
#   Non-triton reference (from lightllm) - tensor reduction operation
# ============================================================================


# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, routed_scaling_factor):
        """
        Args:
            input_tensor: [token_num, topk_num, hidden_dim]
            routed_scaling_factor: float scalar
        Returns:
            output: [token_num, hidden_dim]
        """
        assert input_tensor.is_contiguous()

        token_num, topk_num, hidden_dim = input_tensor.shape
        output = torch.empty(
            (token_num, hidden_dim),
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )

        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 1
        num_warps = 16

        grid = (
            triton.cdiv(token_num, BLOCK_M),
            triton.cdiv(hidden_dim, BLOCK_DIM),
        )

        _moe_sum_reduce_kernel[grid](
            input_tensor,
            *input_tensor.stride(),
            output,
            *output.stride(),
            token_num=token_num,
            topk_num=topk_num,
            hidden_dim=hidden_dim,
            routed_scaling_factor=routed_scaling_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_DIM=BLOCK_DIM,
            NUM_STAGE=NUM_STAGE,
            num_warps=num_warps,
        )
        return output


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, input_tensor, routed_scaling_factor):
        """
        Reference implementation using simple torch operations
        """
        # Sum across topk dimension and apply scaling factor
        output = input_tensor.sum(dim=1) * routed_scaling_factor
        return output


def get_inputs():
    # Example dimensions for MOE sum reduce
    token_num = 256
    topk_num = 8
    hidden_dim = 4096
    dtype = torch.float16

    input_tensor = torch.randn(
        (token_num, topk_num, hidden_dim),
        dtype=dtype,
        
    )
    routed_scaling_factor = 1.0 / topk_num

    return [input_tensor, routed_scaling_factor]


def get_init_inputs():
    return []
