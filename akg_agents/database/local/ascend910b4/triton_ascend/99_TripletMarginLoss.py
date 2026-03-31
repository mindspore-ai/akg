import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 4, 'BLOCK_SIZE_D': 1024}),
        triton.Config({'BLOCK_SIZE_B': 2, 'BLOCK_SIZE_D': 2048}),
    ],
    key=['B', 'D'],
)
@triton.jit
def aikg_99_TripletMarginLoss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    output_ptr,
    per_sample_loss_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    margin: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton Triplet Margin Loss 内核
    计算 triplet margin loss = mean(max(||anchor - positive|| - ||anchor - negative|| + margin, 0))
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    
    # 第一阶段：计算每个样本的triplet loss
    for b_start in range(pid * BLOCK_SIZE_B, B, num_cores * BLOCK_SIZE_B):
        b_end = min(b_start + BLOCK_SIZE_B, B)
        current_block_size = b_end - b_start
        
        # 初始化累加器
        dist_pos_acc = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
        dist_neg_acc = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
        
        # 计算欧氏距离平方
        for d_start in range(0, D, BLOCK_SIZE_D):
            d_end = min(d_start + BLOCK_SIZE_D, D)
            current_d_size = d_end - d_start
            
            # 创建偏移掩码
            b_offsets = tl.arange(0, BLOCK_SIZE_B)
            d_offsets = tl.arange(0, BLOCK_SIZE_D)
            
            # 计算全局偏移
            anchor_offsets = (b_start + b_offsets[:, None]) * D + (d_start + d_offsets[None, :])
            positive_offsets = (b_start + b_offsets[:, None]) * D + (d_start + d_offsets[None, :])
            negative_offsets = (b_start + b_offsets[:, None]) * D + (d_start + d_offsets[None, :])
            
            # 创建掩码
            b_mask = (b_start + b_offsets) < B
            d_mask = (d_start + d_offsets) < D
            full_mask = b_mask[:, None] & d_mask[None, :]
            
            # 加载数据
            anchor_block = tl.load(anchor_ptr + anchor_offsets, mask=full_mask, other=0.0)
            positive_block = tl.load(positive_ptr + positive_offsets, mask=full_mask, other=0.0)
            negative_block = tl.load(negative_ptr + negative_offsets, mask=full_mask, other=0.0)
            
            # 计算差值平方并直接累加，避免创建大临时缓冲区
            diff_pos = anchor_block - positive_block
            diff_neg = anchor_block - negative_block
            
            # 直接累加平方值，避免存储中间结果
            dist_pos_acc += tl.sum(diff_pos * diff_pos, axis=1)
            dist_neg_acc += tl.sum(diff_neg * diff_neg, axis=1)
        
        # 计算L2距离
        dist_pos = tl.sqrt(dist_pos_acc)
        dist_neg = tl.sqrt(dist_neg_acc)
        
        # 计算triplet loss
        temp = dist_pos - dist_neg + margin
        loss_tile = tl.maximum(temp, 0.0)
        
        # 存储每个样本的损失
        loss_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
        loss_mask = loss_offsets < B
        tl.store(per_sample_loss_ptr + loss_offsets, loss_tile, mask=loss_mask)
    
    # 第二阶段：归约计算batch平均损失
    if pid == 0:
        total_loss = 0.0
        
        for b_start in range(0, B, BLOCK_SIZE_B):
            b_end = min(b_start + BLOCK_SIZE_B, B)
            
            # 加载样本损失
            loss_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
            loss_mask = loss_offsets < B
            loss_tile = tl.load(per_sample_loss_ptr + loss_offsets, mask=loss_mask, other=0.0)
            
            # 累加
            total_loss += tl.sum(loss_tile)
        
        # 计算平均值并存储
        avg_loss = total_loss / B
        tl.store(output_ptr, avg_loss.to(tl.float32))


def aikg_99_TripletMarginLoss_triton_ascend_torch(anchor, positive, negative, margin=1.0):
    """
    Triton Triplet Margin Loss 启动函数
    
    参数:
        anchor: 锚点样本张量 [B, D]  # B=128, D=4096
        positive: 正样本张量 [B, D]  
        negative: 负样本张量 [B, D]
        margin: 边界值
    
    返回:
        triplet margin loss 标量
    """
    # 获取输入形状
    B, D = anchor.shape
    
    # 确保输入是连续的
    if not anchor.is_contiguous():
        anchor = anchor.contiguous()
    if not positive.is_contiguous():
        positive = positive.contiguous()
    if not negative.is_contiguous():
        negative = negative.contiguous()
    
    # 分配输出张量
    output = torch.empty((1,), dtype=torch.float32, device=anchor.device)
    
    # 分配中间结果张量（每个样本的损失）
    per_sample_loss = torch.empty((B,), dtype=torch.float32, device=anchor.device)
    
    # 使用固定核心数启动（Ascend 910B4有20个AI Core）
    num_cores = 20
    
    # 启动内核
    aikg_99_TripletMarginLoss_kernel[(num_cores,)](
        anchor,
        positive,
        negative,
        output,
        per_sample_loss,
        B=B,
        D=D,
        margin=margin,
        # BLOCK_SIZE_B 和 BLOCK_SIZE_D 由 autotune 自动传入
    )
    
    return output[0]