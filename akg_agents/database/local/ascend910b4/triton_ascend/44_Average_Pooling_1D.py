import torch
import triton
import triton.language as tl

@triton.jit
def aikg_44_Average_Pooling_1D_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    B,      # batch_size
    C,      # in_channels
    L_in,   # 输入长度
    L_out,  # 输出长度
    kernel_size,  # 池化核大小
    stride,       # 步长
    padding,      # 填充
    TILE_SIZE: tl.constexpr,  # tile大小
):
    """
    1D平均池化Triton内核
    每个程序处理一个通道的一个tile
    """
    # 获取程序ID
    pid_b = tl.program_id(0)  # batch维度
    pid_c = tl.program_id(1)  # channel维度
    pid_tile = tl.program_id(2)  # tile维度
    
    # 计算当前tile的起始位置
    l_out_start = pid_tile * TILE_SIZE
    l_out_end = min(l_out_start + TILE_SIZE, L_out)
    current_tile_size = l_out_end - l_out_start
    
    # 检查边界
    if pid_b >= B or pid_c >= C:
        return
    
    # 计算输入张量的基础偏移
    x_base = pid_b * C * L_in + pid_c * L_in
    y_base = pid_b * C * L_out + pid_c * L_out
    
    # 处理当前tile
    for pos in range(current_tile_size):
        l_out_idx = l_out_start + pos
        
        # 计算输入窗口起始位置
        l_in_start = l_out_idx * stride - padding
        l_in_end = l_in_start + kernel_size
        
        # 初始化累加器
        sum_val = 0.0
        count = 0
        
        # 遍历池化窗口
        for k in range(kernel_size):
            l_in_idx = l_in_start + k
            
            # 检查边界并加载数据
            if l_in_idx >= 0 and l_in_idx < L_in:
                # 加载有效数据
                x_val = tl.load(x_ptr + x_base + l_in_idx)
                sum_val += x_val
                count += 1
            # 边界外填充0（AvgPool1d默认行为）
        
        # 计算平均值
        if count > 0:
            avg_val = sum_val / kernel_size  # 使用kernel_size而不是count（torch.nn.AvgPool1d的行为）
        else:
            avg_val = 0.0
        
        # 存储结果
        tl.store(y_ptr + y_base + l_out_idx, avg_val)


def aikg_44_Average_Pooling_1D_triton_ascend_torch(x):
    """
    1D平均池化Triton启动函数
    
    Args:
        x: 输入张量，形状为 (batch_size, in_channels, input_length)
        
    Returns:
        输出张量，形状为 (batch_size, in_channels, output_length)
    """
    # 从输入张量获取shape参数
    B, C, L_in = x.shape  # batch_size=16, in_channels=32, input_length=128
    
    # 固定参数（从任务描述中获取）
    kernel_size = 4
    stride = 2
    padding = 1
    
    # 计算输出长度
    L_out = (L_in + 2 * padding - kernel_size) // stride + 1
    
    # 分配输出张量
    y = torch.empty((B, C, L_out), dtype=x.dtype, device=x.device)
    
    # 设置tile大小
    TILE_SIZE = 32
    
    # 计算grid大小
    num_tiles = (L_out + TILE_SIZE - 1) // TILE_SIZE
    grid = (B, C, num_tiles)
    
    # 启动内核
    aikg_44_Average_Pooling_1D_kernel[grid](
        x, y, B, C, L_in, L_out, kernel_size, stride, padding,
        TILE_SIZE=TILE_SIZE
    )
    
    return y