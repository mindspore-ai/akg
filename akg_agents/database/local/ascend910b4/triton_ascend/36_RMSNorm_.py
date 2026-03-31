import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_F': 16, 'BLOCK_SIZE_D1D2': 256}),
        triton.Config({'BLOCK_SIZE_F': 32, 'BLOCK_SIZE_D1D2': 128}),
        triton.Config({'BLOCK_SIZE_F': 8, 'BLOCK_SIZE_D1D2': 512}),
        triton.Config({'BLOCK_SIZE_F': 16, 'BLOCK_SIZE_D1D2': 512}),
        triton.Config({'BLOCK_SIZE_F': 32, 'BLOCK_SIZE_D1D2': 256}),
    ],
    key=['B', 'F', 'D1', 'D2'],
)
@triton.jit
def aikg_36_RMSNorm__kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    F: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
    BLOCK_SIZE_D1D2: tl.constexpr,
):
    """
    RMSNorm Triton 内核实现
    每个程序处理一个batch和D1D2块
    """
    # 获取程序ID
    pid_b = tl.program_id(0)
    pid_d1d2 = tl.program_id(1)
    
    # 计算总D1D2元素数
    total_d1d2 = D1 * D2
    
    # 计算当前D1D2块的偏移和掩码
    d1d2_start = pid_d1d2 * BLOCK_SIZE_D1D2
    d1d2_offsets = d1d2_start + tl.arange(0, BLOCK_SIZE_D1D2)
    d1d2_mask = d1d2_offsets < total_d1d2
    
    # 初始化RMS累加器
    rms_accum = tl.zeros((BLOCK_SIZE_D1D2,), dtype=tl.float32)
    
    # 第一阶段：计算平方和
    num_blocks_f = tl.cdiv(F, BLOCK_SIZE_F)
    
    for f_block in range(num_blocks_f):
        # 计算当前F块的偏移和掩码
        f_start = f_block * BLOCK_SIZE_F
        f_offsets = f_start + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < F
        
        # 计算输入指针偏移
        x_offset_base = pid_b * F * total_d1d2
        x_offset_f = f_offsets[:, None] * total_d1d2
        x_offset_d1d2 = d1d2_offsets[None, :]
        x_offsets = x_offset_base + x_offset_f + x_offset_d1d2
        
        # 加载输入数据
        load_mask = f_mask[:, None] & d1d2_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0)
        
        # 计算平方并累加到rms_accum
        x_square = x_tile * x_tile
        rms_accum += tl.sum(x_square, axis=0)
    
    # 计算RMS值
    rms_mean = rms_accum / F
    rms_mean_eps = rms_mean + eps
    rms = tl.sqrt(rms_mean_eps)
    
    # 第二阶段：归一化计算
    for f_block in range(num_blocks_f):
        # 计算当前F块的偏移和掩码
        f_start = f_block * BLOCK_SIZE_F
        f_offsets = f_start + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < F
        
        # 计算输入指针偏移
        x_offset_base = pid_b * F * total_d1d2
        x_offset_f = f_offsets[:, None] * total_d1d2
        x_offset_d1d2 = d1d2_offsets[None, :]
        x_offsets = x_offset_base + x_offset_f + x_offset_d1d2
        
        # 计算输出指针偏移
        y_offset_base = pid_b * F * total_d1d2
        y_offset_f = f_offsets[:, None] * total_d1d2
        y_offset_d1d2 = d1d2_offsets[None, :]
        y_offsets = y_offset_base + y_offset_f + y_offset_d1d2
        
        # 加载输入数据
        load_mask = f_mask[:, None] & d1d2_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0)
        
        # 归一化计算
        y_tile = x_tile / rms[None, :]
        
        # 存储结果
        tl.store(y_ptr + y_offsets, y_tile, mask=load_mask)


def aikg_36_RMSNorm__triton_ascend_torch(x: torch.Tensor, eps: float = 1e-5):
    """
    RMSNorm Triton 启动函数
    
    Args:
        x: 输入张量，形状为 [B, F, D1, D2]
        eps: 防止除零的小值，默认为1e-5
        
    Returns:
        归一化后的张量，形状与输入相同
    """
    # 获取输入形状参数
    B, F, D1, D2 = x.shape  # B=16, F=64, D1=256, D2=256
    
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算网格大小
    total_d1d2 = D1 * D2
    grid = lambda meta: (B, triton.cdiv(total_d1d2, meta['BLOCK_SIZE_D1D2']))
    
    # 启动内核
    aikg_36_RMSNorm__kernel[grid](
        x, y, B, F, D1, D2, eps,
        # BLOCK_SIZE_F 和 BLOCK_SIZE_D1D2 由 autotune 自动传入
    )
    
    return y