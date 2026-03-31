import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_CORES': 40}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_CORES': 40}),
    ],
    key=['B', 'D']
)
@triton.jit
def aikg_97_CosineSimilarityLoss_kernel(
    predictions_ptr,  # [B, D] f16
    targets_ptr,      # [B, D] f16
    output_ptr,       # [1] f32
    B: tl.constexpr,  # batch size
    D: tl.constexpr,  # feature dimension
    stride_pred_b: tl.constexpr,
    stride_pred_d: tl.constexpr,
    stride_targ_b: tl.constexpr,
    stride_targ_d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """
    Cosine Similarity Loss Kernel
    计算余弦相似度损失: mean(1 - cosine_similarity(predictions, targets, dim=1))
    """
    # 获取核心ID
    core_id = tl.program_id(0)
    
    # 计算每个核心处理的batch数量
    b_per_core = tl.cdiv(B, NUM_CORES)
    b_start = core_id * b_per_core
    b_end = min(b_start + b_per_core, B)
    
    # 初始化当前核心的损失累加器
    core_loss_acc = 0.0
    
    # 处理当前核心负责的batch
    for pid in range(b_start, b_end):
        # 初始化当前样本的累加器
        dot_acc = 0.0
        pred_norm_acc = 0.0
        target_norm_acc = 0.0
        
        # 分块处理特征维度
        for start in range(0, D, BLOCK_SIZE):
            # 计算当前块的偏移和掩码
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < D
            
            # 加载预测数据
            pred_offset = pid * stride_pred_b + offsets * stride_pred_d
            pred_tile = tl.load(predictions_ptr + pred_offset, mask=mask, other=0.0)
            
            # 加载目标数据
            target_offset = pid * stride_targ_b + offsets * stride_targ_d
            target_tile = tl.load(targets_ptr + target_offset, mask=mask, other=0.0)
            
            # 转换为f32精度
            pred_f32 = pred_tile.to(tl.float32)
            target_f32 = target_tile.to(tl.float32)
            
            # 计算点积和模长平方
            dot_product = pred_f32 * target_f32
            pred_square = pred_f32 * pred_f32
            target_square = target_f32 * target_f32
            
            # 累加结果
            dot_acc += tl.sum(dot_product, axis=0)
            pred_norm_acc += tl.sum(pred_square, axis=0)
            target_norm_acc += tl.sum(target_square, axis=0)
        
        # 计算模长
        pred_norm = tl.sqrt(pred_norm_acc)
        target_norm = tl.sqrt(target_norm_acc)
        
        # 计算余弦相似度
        norm_product = pred_norm * target_norm
        
        # 直接比较标量值（避免使用tl.get_element）
        if norm_product == 0.0:
            cosine_sim = dot_acc
        else:
            cosine_sim = dot_acc / norm_product
        
        # 计算1 - cosine_similarity
        loss_per_sample = 1.0 - cosine_sim
        
        # 累加到核心损失
        core_loss_acc += loss_per_sample
    
    # 原子操作累加到全局输出（确保类型匹配）
    tl.atomic_add(output_ptr, core_loss_acc.to(tl.float32))


def aikg_97_CosineSimilarityLoss_triton_ascend_torch(predictions, targets):
    """
    Cosine Similarity Loss Triton实现
    
    参数:
        predictions: torch.Tensor [B, D] f16 - 预测向量
        targets: torch.Tensor [B, D] f16 - 目标向量
        
    返回:
        torch.Tensor [1] f32 - 平均余弦相似度损失
    """
    # 确保输入张量是连续的
    if not predictions.is_contiguous():
        predictions = predictions.contiguous()
    if not targets.is_contiguous():
        targets = targets.contiguous()
    
    # 获取输入形状
    B, D = predictions.shape  # B=128, D=4096
    
    # 分配输出张量（初始化为0）
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # 定义启动网格的lambda函数
    grid = lambda meta: (meta['NUM_CORES'],)
    
    # 启动内核
    aikg_97_CosineSimilarityLoss_kernel[grid](
        predictions, targets, output,
        B, D,
        predictions.stride(0), predictions.stride(1),
        targets.stride(0), targets.stride(1),
        # BLOCK_SIZE和NUM_CORES由autotune自动传入
    )
    
    # 计算平均损失
    mean_loss = output[0] / B
    output[0] = mean_loss
    
    return output