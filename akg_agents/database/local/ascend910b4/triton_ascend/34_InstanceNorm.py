import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'TILE_H': 32, 'TILE_W': 32}),
        triton.Config({'TILE_H': 64, 'TILE_W': 64}),
    ],
    key=['H', 'W'],
)
@triton.jit
def aikg_34_InstanceNorm_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    B,      # batch_size
    C,      # num_features
    H,      # height
    W,      # width
    eps,    # epsilon
    stride_b, stride_c, stride_h, stride_w,  # 输入张量步长
    TILE_H: tl.constexpr,  # 高度分块大小
    TILE_W: tl.constexpr,  # 宽度分块大小
):
    """
    Instance Normalization Triton 内核
    每个程序处理一个通道 (b, c)
    """
    # 获取当前程序处理的批次和通道
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # 计算当前通道的基址偏移
    base_offset = pid_b * stride_b + pid_c * stride_c
    
    # 初始化累加器
    mean_acc = 0.0
    var_acc = 0.0
    total_elements = H * W
    
    # 第一遍：计算均值和方差
    for h_outer in range(0, H, TILE_H):
        for w_outer in range(0, W, TILE_W):
            # 计算当前块的偏移
            h_offsets = h_outer + tl.arange(0, TILE_H)
            w_offsets = w_outer + tl.arange(0, TILE_W)
            
            # 创建2D掩码
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            mask_2d = h_mask[:, None] & w_mask[None, :]
            
            # 计算内存偏移
            offsets_2d = base_offset + h_offsets[:, None] * stride_h + w_offsets[None, :] * stride_w
            
            # 加载数据
            x_tile = tl.load(x_ptr + offsets_2d, mask=mask_2d, other=0.0)
            
            # 累加求和
            tile_sum = tl.sum(x_tile)
            mean_acc += tile_sum
            
            # 累加平方和
            x_squared = x_tile * x_tile
            tile_sq_sum = tl.sum(x_squared)
            var_acc += tile_sq_sum
    
    # 计算最终均值和方差
    mean_val = mean_acc / total_elements
    var_val = var_acc / total_elements - mean_val * mean_val
    std_val = tl.sqrt(var_val + eps)
    
    # 第二遍：归一化
    for h_outer in range(0, H, TILE_H):
        for w_outer in range(0, W, TILE_W):
            # 计算当前块的偏移
            h_offsets = h_outer + tl.arange(0, TILE_H)
            w_offsets = w_outer + tl.arange(0, TILE_W)
            
            # 创建2D掩码
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            mask_2d = h_mask[:, None] & w_mask[None, :]
            
            # 计算内存偏移
            offsets_2d = base_offset + h_offsets[:, None] * stride_h + w_offsets[None, :] * stride_w
            
            # 加载数据
            x_tile = tl.load(x_ptr + offsets_2d, mask=mask_2d, other=0.0)
            
            # 归一化: (x - mean) / std
            y_tile = (x_tile - mean_val) / std_val
            
            # 存储结果
            tl.store(y_ptr + offsets_2d, y_tile, mask=mask_2d)


def aikg_34_InstanceNorm_triton_ascend_torch(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Instance Normalization Triton 启动函数
    
    Args:
        x: 输入张量，形状为 (batch_size, num_features, height, width)
        eps: 数值稳定性参数
        
    Returns:
        归一化后的输出张量，形状与输入相同
    """
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入形状
    B, C, H, W = x.shape  # batch_size=16, features=64, height=256, width=256
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算网格大小
    grid = (B, C)
    
    # 获取步长信息
    stride_b, stride_c, stride_h, stride_w = x.stride()
    
    # 启动内核
    aikg_34_InstanceNorm_kernel[grid](
        x, y, B, C, H, W, eps,
        stride_b, stride_c, stride_h, stride_w
    )
    
    return y