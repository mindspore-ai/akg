import torch
import triton
import triton.language as tl

@triton.jit
def aikg_100_HingeLoss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """
    Hinge Loss 内核实现
    计算: mean(max(0, 1 - predictions * targets))
    """
    # 获取核心ID
    pid = tl.program_id(0)
    
    # 计算总块数
    NUM_BLOCKS_B = tl.cdiv(B, BLOCK_SIZE_B)
    NUM_BLOCKS = NUM_BLOCKS_B
    
    # 初始化累加器
    total_sum = 0.0
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        # 计算当前块的起始位置
        b_start = block_idx * BLOCK_SIZE_B
        b_end = min(b_start + BLOCK_SIZE_B, B)
        
        # 当前块的实际大小
        current_block_size = b_end - b_start
        
        # 处理当前块的所有元素
        for d in range(0, D, BLOCK_SIZE_D):
            # 创建偏移量
            b_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
            d_offsets = d + tl.arange(0, BLOCK_SIZE_D)
            
            # 创建掩码
            b_mask = b_offsets < B
            d_mask = d_offsets < D
            full_mask = b_mask[:, None] & d_mask[None, :]
            
            # 计算内存偏移
            offsets = b_offsets[:, None] * D + d_offsets[None, :]
            
            # 加载数据
            predictions = tl.load(predictions_ptr + offsets, mask=full_mask, other=0.0)
            targets = tl.load(targets_ptr + offsets, mask=full_mask, other=0.0)
            
            # 计算 Hinge Loss: max(0, 1 - predictions * targets)
            product = predictions * targets
            hinge = 1.0 - product
            hinge_clamped = tl.where(hinge > 0.0, hinge, 0.0)
            
            # 累加到总和
            block_sum = tl.sum(hinge_clamped)
            total_sum += block_sum
    
    # 原子操作累加到全局输出
    tl.atomic_add(output_ptr, total_sum)


def aikg_100_HingeLoss_triton_ascend_torch(predictions, targets):
    """
    Hinge Loss Triton 启动函数
    
    参数:
        predictions: 预测值张量, shape [B, D]
        targets: 目标值张量, shape [B, D], 值为 -1 或 1
    
    返回:
        hinge_loss: 标量损失值
    """
    # 获取输入形状
    B, D = predictions.shape  # B=128, D=1
    
    # 确保输入张量是连续的
    if not predictions.is_contiguous():
        predictions = predictions.contiguous()
    if not targets.is_contiguous():
        targets = targets.contiguous()
    
    # 创建输出张量（初始化为0）
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # 使用lambda函数定义grid，autotune会自动选择最佳配置
    NUM_CORES = 40
    grid = (NUM_CORES,)
    
    aikg_100_HingeLoss_kernel[grid](
        predictions, targets, output,
        B, D,
        BLOCK_SIZE_B=128, BLOCK_SIZE_D=1, NUM_CORES=NUM_CORES,
    )
    
    # 计算平均值
    total_elements = B * D
    hinge_loss = output.item() / total_elements
    
    return torch.tensor(hinge_loss, device=predictions.device)