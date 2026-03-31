import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384, 'VEC_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 8192, 'VEC_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_SIZE': 64}),
    ],
    key=['N'],
)
@triton.jit
def aikg_22_Tanh_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Triton Tanh 激活函数内核
    每个程序处理一个数据块
    
    Args:
        x_ptr: 输入张量指针
        y_ptr: 输出张量指针
        N: 输入张量总元素数
        BLOCK_SIZE: 每个程序处理的元素数
        VEC_SIZE: 向量化处理的大小
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前程序处理的数据偏移
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建边界掩码
    mask = offsets < N
    
    # 从全局内存加载数据到UB
    x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 使用向量化计算tanh
    # Triton内置的tanh函数会自动向量化处理
    y_data = tl.tanh(x_data)
    
    # 将结果写回全局内存
    tl.store(y_ptr + offsets, y_data, mask=mask)


def aikg_22_Tanh_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Triton Tanh 激活函数启动函数
    
    Args:
        x (torch.Tensor): 输入张量，任意形状
        
    Returns:
        torch.Tensor: 输出张量，与输入形状相同，应用了Tanh激活
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入张量的总元素数
    N = x.numel()
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 定义grid大小计算函数
    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_22_Tanh_kernel[grid](
        x, y, N
        # BLOCK_SIZE和VEC_SIZE由autotune自动传入
    )
    
    return y


# 测试代码
if __name__ == "__main__":
    # 创建测试输入
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    
    # 运行Triton实现
    output = aikg_22_Tanh_triton_ascend_torch(x)
    
    # 验证结果
    expected = torch.tanh(x)
    print("结果一致:", torch.allclose(output, expected, atol=1e-6))