import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 不同的核数配置
        triton.Config({'NUM_H_CORES': 1}),   # 单核baseline
        triton.Config({'NUM_H_CORES': 4}),   # 4核，每核32行
        triton.Config({'NUM_H_CORES': 8}),   # 8核，每核16行
        triton.Config({'NUM_H_CORES': 16}),  # 16核，每核8行
        triton.Config({'NUM_H_CORES': 32}),  # 32核，每核4行
        triton.Config({'NUM_H_CORES': 64}),  # 64核，每核2行
        triton.Config({'NUM_H_CORES': 128}), # 128核，每核1行
    ],
    key=['H', 'W'],
)
@triton.jit
def broadcast_kernel_parallel(
    input_ptr,
    output_ptr,
    stride_input_h,
    stride_output_h,
    stride_output_w,
    H: tl.constexpr,
    W: tl.constexpr,
    NUM_H_CORES: tl.constexpr,
):
    """
    多核并行Broadcast kernel: (1, H, 1) -> (1, H, W)
    
    每个核负责几行H，并行处理
    """
    pid = tl.program_id(0)
    
    # 计算每个核负责的H范围
    h_per_core = (H + NUM_H_CORES - 1) // NUM_H_CORES
    h_start = pid * h_per_core
    h_end = tl.minimum(h_start + h_per_core, H)

    num_h = tl.minimum(h_per_core, H - h_start)
    
    if num_h > 0:
        # 加载当前核负责的H行
        offs_h_local = tl.arange(0, 128)  # 最多128行
        mask_h = (h_start + offs_h_local) < h_end
        mask_h = mask_h & (offs_h_local < num_h)
        
        offs_h_global = h_start + offs_h_local
        input_vals = tl.load(input_ptr + offs_h_global * stride_input_h, 
                            mask=mask_h, other=0.0)
        
        # W维度偏移
        offs_w = tl.arange(0, W)
        
        # 计算2D输出偏移：(num_h, W)
        output_offs = (offs_h_global[:, None] * stride_output_h + 
                      offs_w[None, :] * stride_output_w)
        
        # 创建2D mask
        mask_2d = mask_h[:, None]
        
        # 一次性存储：input_vals[:, None]自动broadcast到(num_h, W)
        tl.store(output_ptr + output_offs, input_vals[:, None], mask=mask_2d)


@triton.autotune(
    configs=[
        triton.Config({'NUM_CORES': 40, 'SUB_B': 12, 'SUB_HW': 1024}),   # 1701.14us，最优，用满物理核
        triton.Config({'NUM_CORES': 128, 'SUB_B': 12, 'SUB_HW': 1024}),   # 1741.76us，核数>40，调度开销大
    ],
    key=['B', 'HW'],
)
@triton.jit
def div_flatten_kernel(
    input1_ptr,
    input2_ptr,
    output_ptr,
    B, HW,
    stride_input1_b, stride_input1_hw,
    stride_input2_hw,
    stride_output_b, stride_output_hw,
    NUM_CORES: tl.constexpr,
    SUB_B: tl.constexpr,
    SUB_HW: tl.constexpr,
):
    """
    除法kernel（处理已经broadcast+flatten好的数据）
    """
    pid = tl.program_id(0)
    
    b_per_core = (B + NUM_CORES - 1) // NUM_CORES
    b_start = pid * b_per_core
    b_end = tl.minimum(b_start + b_per_core, B)
    
    for b_block in range(b_start, b_end, SUB_B):
        offs_b = b_block + tl.arange(0, SUB_B)
        mask_b = offs_b < b_end
        
        for hw_block in range(0, HW, SUB_HW):
            offs_hw = hw_block + tl.arange(0, SUB_HW)
            mask_hw = offs_hw < HW
            
            input2 = tl.load(input2_ptr + offs_hw * stride_input2_hw, 
                           mask=mask_hw, other=1.0)
            
            input1_offs = (offs_b[:, None] * stride_input1_b +
                          offs_hw[None, :] * stride_input1_hw)
            output_offs = (offs_b[:, None] * stride_output_b +
                          offs_hw[None, :] * stride_output_hw)
            
            mask_2d = mask_b[:, None] & mask_hw[None, :]
            
            input1 = tl.load(input1_ptr + input1_offs, mask=mask_2d, other=0.0)
            output = input1 / input2[None, :]
            tl.store(output_ptr + output_offs, output, mask=mask_2d)


def custom_op_triton_torch(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    优化版两阶段kernel：多核并行broadcast
    """
    assert input1.ndim == 3 and input2.ndim == 3
    assert input2.shape[0] == 1 and input2.shape[2] == 1
    assert input1.shape[1] == input2.shape[1]
    assert input1.dtype == input2.dtype
    
    B, H, W = input1.shape
    HW = H * W
    
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    
    # ============ 阶段1：多核并行Broadcast（autotune优化）============
    input2_broadcast = torch.empty(1, H, W, dtype=input2.dtype, device=input2.device)
    
    # Grid由autotune的NUM_H_CORES决定
    grid_broadcast = lambda meta: (meta['NUM_H_CORES'],)
    broadcast_kernel_parallel[grid_broadcast](
        input2, input2_broadcast,
        input2.stride(1),
        input2_broadcast.stride(1), input2_broadcast.stride(2),
        H=H, W=W,
    )
    
    # ============ 阶段2：Division kernel ============
    input1_flat = input1.reshape(B, HW).contiguous()
    input2_flat = input2_broadcast.reshape(1, HW).contiguous()
    output_flat = torch.empty(B, HW, dtype=input1.dtype, device=input1.device)
    
    grid_div = lambda meta: (meta['NUM_CORES'],)
    div_flatten_kernel[grid_div](
        input1_flat, input2_flat, output_flat,
        B, HW,
        input1_flat.stride(0), input1_flat.stride(1),
        input2_flat.stride(1),
        output_flat.stride(0), output_flat.stride(1),
    )
    
    output = output_flat.reshape(B, H, W)
    return output