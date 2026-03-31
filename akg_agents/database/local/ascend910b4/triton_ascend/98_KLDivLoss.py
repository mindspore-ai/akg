import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 4096}),
    ],
    key=['B', 'N'],
)
@triton.jit
def aikg_98_KLDivLoss_kernel(
    predictions_ptr,  # 输入预测分布 [B, N]
    targets_ptr,      # 输入目标分布 [B, N]
    batch_results_ptr,  # 各batch的KL散度结果 [B]
    B: tl.constexpr,  # batch size
    N: tl.constexpr,  # 特征维度
    EPS: tl.constexpr,  # 防止log(0)的小值
    BLOCK_SIZE_N: tl.constexpr,  # N维度分块大小
):
    """
    KL散度计算内核：target * (log(target) - log(prediction))
    第一阶段：每个batch独立计算KL散度总和
    """
    # 获取当前程序ID（处理哪个batch）
    pid_b = tl.program_id(0)
    
    # 检查是否超出batch范围
    if pid_b >= B:
        return
    
    # 初始化当前batch的累加器（标量）
    batch_acc = 0.0
    
    # 计算当前batch的起始指针
    pred_batch_start = pid_b * N
    target_batch_start = pid_b * N
    
    # 分块处理特征维度N
    for start_n in range(0, N, BLOCK_SIZE_N):
        # 计算当前块的偏移
        offsets = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        
        # 加载预测和目标数据
        pred_tile = tl.load(predictions_ptr + pred_batch_start + offsets, mask=mask, other=0.0)
        target_tile = tl.load(targets_ptr + target_batch_start + offsets, mask=mask, other=0.0)
        
        # 防止log(0) - 使用EPS作为最小值
        pred_tile = tl.maximum(pred_tile, EPS)
        target_tile = tl.maximum(target_tile, EPS)
        
        # 计算log值
        log_pred = tl.log(pred_tile)
        log_target = tl.log(target_tile)
        
        # 计算KL散度元素：target * (log(target) - log(prediction))
        diff = log_target - log_pred
        kl_element = target_tile * diff
        
        # 累加当前块的结果
        tile_sum = tl.sum(kl_element, axis=0)
        batch_acc += tile_sum
    
    # 存储当前batch的结果到全局内存（标量存储）
    tl.store(batch_results_ptr + pid_b, batch_acc)


@triton.jit
def aikg_98_KLDivLoss_reduce_kernel(
    batch_results_ptr,  # 各batch的KL散度结果 [B]
    output_ptr,         # 最终输出 [1]
    B: tl.constexpr,    # batch size
):
    """
    第二阶段：batchmean归约内核（单核处理）
    """
    # 只有第一个程序执行归约
    if tl.program_id(0) != 0:
        return
    
    # 初始化最终累加器
    final_acc = 0.0
    
    # 累加所有batch的结果
    for b_idx in range(B):
        batch_val = tl.load(batch_results_ptr + b_idx)
        final_acc += batch_val
    
    # 计算batch平均值
    batch_mean = final_acc / B
    
    # 存储最终结果
    tl.store(output_ptr, batch_mean)


def aikg_98_KLDivLoss_triton_ascend_torch(predictions, targets):
    """
    KL散度计算启动函数
    
    Args:
        predictions: 预测分布张量 [batch_size, feature_dim]
        targets: 目标分布张量 [batch_size, feature_dim]
        
    Returns:
        output: KL散度标量 [1]
    """
    # 检查输入形状
    assert predictions.shape == targets.shape, "预测和目标形状必须相同"
    assert predictions.dim() == 2, "输入必须是2D张量"
    
    # 获取形状参数
    B, N = predictions.shape  # B=128, N=4096
    
    # 确保输入是连续的
    if not predictions.is_contiguous():
        predictions = predictions.contiguous()
    if not targets.is_contiguous():
        targets = targets.contiguous()
    
    # 分配中间结果张量（各batch的KL散度）
    batch_results = torch.empty(B, dtype=torch.float32, device=predictions.device)
    
    # 分配输出张量
    output = torch.empty(1, dtype=torch.float32, device=predictions.device)
    
    # 定义常量
    EPS = 1e-8
    
    # 启动第一阶段内核：每个batch独立计算KL散度
    grid_stage1 = (B,)  # 每个batch一个程序
    
    aikg_98_KLDivLoss_kernel[grid_stage1](
        predictions, targets, batch_results,
        B, N, EPS,
        # BLOCK_SIZE_N 由autotune自动选择
    )
    
    # 启动第二阶段内核：batchmean归约
    grid_stage2 = (1,)  # 单核处理归约
    aikg_98_KLDivLoss_reduce_kernel[grid_stage2](
        batch_results, output, B
    )
    
    return output