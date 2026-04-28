import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_B和SUB_HW控制B和HW维度的切分
        triton.Config({'BLOCK_B': 2, 'SUB_HW': 16384}), # 最优，核数=32，被shape整除
        triton.Config({'BLOCK_B': 2, 'SUB_HW': 8096}), # HW偏小
        triton.Config({'BLOCK_B': 4, 'SUB_HW': 16384}), # 核数=16，太小
        triton.Config({'BLOCK_B': 1, 'SUB_HW': 16384}), # 核数=64，大于40
    ],
    key=['B', 'H', 'W'],
)
@triton.jit
def custom_op_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, H, W,
    stride_input1_b,
    stride_input2_b,
    stride_output_b,
    BLOCK_B: tl.constexpr,
    SUB_HW: tl.constexpr,
):
    """
    优化的Triton kernel for 3D逐元素乘法with广播
    输入形状: input1=(B, H, W), input2=(B, 1, 1)
    输出形状: output=(B, H, W)
    
    策略：每个kernel处理BLOCK_B个batch，一次性加载这些batch的y值，循环处理展平的HW
    """
    pid = tl.program_id(0)
    
    # 每个kernel处理BLOCK_B个batch
    b_start = pid * BLOCK_B
    
    # 批量加载BLOCK_B个batch的input2值: (BLOCK_B,)
    b_indices = b_start + tl.arange(0, BLOCK_B)
    mask_b = b_indices < B
    input2_offs = b_indices * stride_input2_b
    input2_vals = tl.load(input2_ptr + input2_offs, mask=mask_b, other=0.0)
    
    # HW总数
    HW = H * W
    
    # 循环：遍历HW维度，每次处理SUB_HW个元素
    for hw_block in range(0, HW, SUB_HW):
        # HW维度偏移
        hw_offs = hw_block + tl.arange(0, SUB_HW)
        mask_hw = hw_offs < HW
        
        # 2D mask: (BLOCK_B, SUB_HW)
        mask_2d = mask_b[:, None] & mask_hw[None, :]
        
        # 计算input1的地址: 使用stride访问3D张量的展平HW
        # offset = b * stride_b + hw (hw是H*W的线性索引)
        input1_offs = b_indices[:, None] * stride_input1_b + hw_offs[None, :]
        
        # 计算output的地址
        output_offs = b_indices[:, None] * stride_output_b + hw_offs[None, :]
        
        # 加载input1: (BLOCK_B, SUB_HW)
        input1 = tl.load(input1_ptr + input1_offs, mask=mask_2d, other=0.0)
        
        # 计算 - input2_vals广播到(BLOCK_B, SUB_HW)
        output = input1 * input2_vals[:, None]
        
        # 存储
        tl.store(output_ptr + output_offs, output, mask=mask_2d)


def custom_op_triton_torch(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Triton实现for (B, H, W) * (B, 1, 1) 广播
    只在B维度切分，通过stride访问展平的HW
    """
    assert input1.ndim == 3 and input2.ndim == 3, "Both inputs must be 3D"
    assert input2.shape[1] == 1 and input2.shape[2] == 1, "input2 must be (B, 1, 1)"
    assert input1.shape[0] == input2.shape[0], "Batch size must match"
    assert input1.dtype == input2.dtype, "Dtypes must match"
    
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    output = torch.empty_like(input1)
    
    B, H, W = input1.shape  # 64, 128, 128
    
    # Grid: 根据BLOCK_B计算需要多少个kernel
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    
    custom_op_kernel[grid](
        input1, input2, output,
        B, H, W,
        input1.stride(0),
        input2.stride(0),
        output.stride(0),
    )
    
    return output