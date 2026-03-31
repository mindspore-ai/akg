import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['N'],
)
@triton.jit
def aikg_96_HuberLoss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    N: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Huber Loss (Smooth L1 Loss) Triton 内核实现
    使用kernel内循环策略，每个program处理多个块，避免UB内存溢出
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # 计算每个program需要处理的元素总数
    ELEMENTS_PER_PROGRAM = (N + num_programs - 1) // num_programs
    
    # 计算当前program负责的数据范围
    program_start = pid * ELEMENTS_PER_PROGRAM
    program_end = min(program_start + ELEMENTS_PER_PROGRAM, N)
    
    # 初始化当前program的累加器
    program_sum = 0.0
    
    # 分块处理当前program负责的数据
    for block_start in range(program_start, program_end, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, program_end)
        actual_size = block_end - block_start
        
        # 创建偏移量和掩码
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < program_end
        
        # 加载预测值和目标值
        pred_tile = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        target_tile = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        # 计算差值
        diff_tile = pred_tile - target_tile
        
        # 计算绝对值
        abs_tile = tl.abs(diff_tile)
        
        # Huber Loss 计算
        # 分支1: 0.5 * diff^2 / beta (当 |diff| < beta)
        branch1_result = 0.5 * diff_tile * diff_tile / beta
        
        # 分支2: |diff| - 0.5 * beta (当 |diff| >= beta)
        branch2_result = abs_tile - 0.5 * beta
        
        # 条件选择
        condition = abs_tile < beta
        result_tile = tl.where(condition, branch1_result, branch2_result)
        
        # 计算当前块的和
        block_sum = tl.sum(result_tile, axis=0)
        
        # 累加到program累加器
        program_sum += block_sum
    
    # 原子累加到全局输出
    tl.atomic_add(output_ptr, program_sum)


def aikg_96_HuberLoss_triton_ascend_torch(predictions, targets, beta=1.0):
    """
    Huber Loss Triton 启动函数
    
    Args:
        predictions: 预测值张量，形状为 [B, D] (B=128, D=4096)
        targets: 目标值张量，形状为 [B, D] (B=128, D=4096)
        beta: Huber Loss 参数，默认为 1.0
    
    Returns:
        loss: 标量损失值
    """
    # 确保输入张量在相同设备上
    assert predictions.device == targets.device, "输入张量必须在相同设备上"
    
    # 展平输入张量
    predictions_flat = predictions.contiguous().view(-1)
    targets_flat = targets.contiguous().view(-1)
    
    # 获取总元素数
    N = predictions_flat.numel()
    
    # 创建输出张量（初始化为0）
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # 计算网格大小 - 使用固定数量的program
    MAX_GRID_SIZE = 128  # 小于65535的安全值
    
    def grid(meta):
        return (min(MAX_GRID_SIZE, triton.cdiv(N, meta['BLOCK_SIZE'])),)
    
    # 启动内核
    aikg_96_HuberLoss_kernel[grid](
        predictions_flat,
        targets_flat,
        output,
        N,
        beta,
    )
    
    # 计算均值并返回
    loss = output[0] / N
    return loss