import torch
import triton
import triton.language as tl

@triton.jit
def aikg_45_Average_Pooling_2D_kernel(
    input_ptr,
    output_ptr,
    B,
    C,
    H,
    W,
    H_out,
    W_out,
    kernel_size,
    stride,
    padding,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Triton 2D平均池化内核
    每个程序处理一个通道的一个tile
    """
    # 将4维grid压缩为3维
    pid = tl.program_id(0)
    
    # 计算各维度索引
    total_tiles_per_channel = tl.cdiv(H_out, BLOCK_H) * tl.cdiv(W_out, BLOCK_W)
    total_tiles = B * C * total_tiles_per_channel
    
    # 计算当前tile的全局索引
    tile_idx = pid
    
    # 计算batch、channel和tile索引
    b_idx = tile_idx // (C * total_tiles_per_channel)
    remainder = tile_idx % (C * total_tiles_per_channel)
    c_idx = remainder // total_tiles_per_channel
    tile_remainder = remainder % total_tiles_per_channel
    
    # 计算height和width tile索引
    tiles_per_width = tl.cdiv(W_out, BLOCK_W)
    h_tile_idx = tile_remainder // tiles_per_width
    w_tile_idx = tile_remainder % tiles_per_width
    
    # 计算当前tile的起始位置
    h_out_start = h_tile_idx * BLOCK_H
    w_out_start = w_tile_idx * BLOCK_W
    
    # 计算当前tile的实际大小
    h_tile = min(BLOCK_H, H_out - h_out_start)
    w_tile = min(BLOCK_W, W_out - w_out_start)
    
    # 处理tile中的每个输出位置
    for h_idx in range(h_tile):
        for w_idx in range(w_tile):
            h_out = h_out_start + h_idx
            w_out = w_out_start + w_idx
            
            # 计算输入窗口的起始位置
            h_start = h_out * stride - padding
            w_start = w_out * stride - padding
            h_end = h_start + kernel_size
            w_end = w_start + kernel_size
            
            # 处理边界条件
            h_start_clamped = max(0, h_start)
            w_start_clamped = max(0, w_start)
            h_end_clamped = min(H, h_end)
            w_end_clamped = min(W, w_end)
            
            # 计算有效窗口区域
            valid_h = h_end_clamped - h_start_clamped
            valid_w = w_end_clamped - w_start_clamped
            valid_area = valid_h * valid_w
            
            # 初始化累加器
            sum_val = 0.0
            
            # 遍历输入窗口进行求和
            for h_in in range(h_start_clamped, h_end_clamped):
                for w_in in range(w_start_clamped, w_end_clamped):
                    # 计算输入偏移
                    input_offset = b_idx * C * H * W + c_idx * H * W + h_in * W + w_in
                    
                    # 加载输入值（带边界检查）
                    # 使用括号拆分链式布尔运算
                    h_in_valid = (h_in >= 0) and (h_in < H)
                    w_in_valid = (w_in >= 0) and (w_in < W)
                    if h_in_valid and w_in_valid:
                        input_val = tl.load(input_ptr + input_offset)
                        sum_val += input_val
            
            # 计算平均值
            if valid_area > 0:
                avg_val = sum_val / valid_area
            else:
                avg_val = 0.0
            
            # 计算输出偏移
            output_offset = b_idx * C * H_out * W_out + c_idx * H_out * W_out + h_out * W_out + w_out
            
            # 存储结果
            tl.store(output_ptr + output_offset, avg_val)


def aikg_45_Average_Pooling_2D_triton_ascend_torch(x, kernel_size=3, stride=None, padding=0):
    """
    Triton 2D平均池化启动函数
    
    Args:
        x: 输入张量，形状为 [B, C, H, W] (batch_size=16, channels=64, height=256, width=256)
        kernel_size: 池化核大小
        stride: 步长，默认为kernel_size
        padding: 填充大小
        
    Returns:
        输出张量，形状为 [B, C, H_out, W_out]
    """
    if stride is None:
        stride = kernel_size
    
    # 获取输入形状
    B, C, H, W = x.shape
    
    # 计算输出形状
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    # 分配输出张量
    output = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # 设置tile大小
    BLOCK_H = 64
    BLOCK_W = 64
    
    # 计算总tile数
    tiles_per_channel = (H_out + BLOCK_H - 1) // BLOCK_H * (W_out + BLOCK_W - 1) // BLOCK_W
    total_tiles = B * C * tiles_per_channel
    
    # 启动内核（使用1维grid）
    aikg_45_Average_Pooling_2D_kernel[(total_tiles,)](
        x, output, B, C, H, W, H_out, W_out, kernel_size, stride, padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return output