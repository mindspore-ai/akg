import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384, 'SUB_BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 8192, 'SUB_BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 4096, 'SUB_BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 2048, 'SUB_BLOCK_SIZE': 128}),
    ],
    key=['total_elements'],
)
@triton.jit
def aikg_94_MSELoss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    """
    Triton MSE Loss 内核
    计算均方误差损失: output = mean((predictions - targets)^2)
    """
    pid = tl.program_id(0)
    
    # 计算当前块的范围
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, total_elements)
    current_block_size = end_idx - start_idx
    
    # 初始化块累加器
    block_sum = 0.0
    
    # 分块处理数据
    for sub_start in range(0, current_block_size, SUB_BLOCK_SIZE):
        sub_end = min(sub_start + SUB_BLOCK_SIZE, current_block_size)
        sub_size = sub_end - sub_start
        
        # 计算当前子块的偏移
        sub_offsets = start_idx + sub_start + tl.arange(0, SUB_BLOCK_SIZE)
        mask = sub_offsets < total_elements
        
        # 加载预测值和目标值
        predictions_tile = tl.load(predictions_ptr + sub_offsets, mask=mask, other=0.0)
        targets_tile = tl.load(targets_ptr + sub_offsets, mask=mask, other=0.0)
        
        # 计算差值: diff = predictions - targets
        diff_tile = predictions_tile - targets_tile
        
        # 计算平方差: sq_diff = diff^2
        sq_diff_tile = diff_tile * diff_tile
        
        # 子块求和
        sub_sum = tl.sum(sq_diff_tile, axis=0)
        block_sum += sub_sum
    
    # 原子操作累加到全局结果
    tl.atomic_add(output_ptr, block_sum)


def aikg_94_MSELoss_triton_ascend_torch(predictions, targets):
    """
    Triton MSE Loss 启动函数
    
    参数:
        predictions: 预测值张量, shape [B, D] = [128, 4096]
        targets: 目标值张量, shape [B, D] = [128, 4096]
    
    返回:
        output: 均方误差损失值, shape [1]
    """
    # 确保输入张量是连续的
    if not predictions.is_contiguous():
        predictions = predictions.contiguous()
    if not targets.is_contiguous():
        targets = targets.contiguous()
    
    # 获取输入形状
    B, D = predictions.shape  # B=128, D=4096
    total_elements = B * D  # 128 * 4096 = 524288
    
    # 分配输出张量并初始化为0
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # 计算网格大小
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_94_MSELoss_kernel[grid](
        predictions,
        targets,
        output,
        total_elements=total_elements,
    )
    
    # 计算均值: output = output / total_elements
    output = output / total_elements
    
    return output