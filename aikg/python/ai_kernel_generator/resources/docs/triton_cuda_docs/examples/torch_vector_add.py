import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # 指向第一个输入向量的指针
               y_ptr,  # 指向第二个输入向量的指针
               output_ptr,  # 指向输出向量的指针
               n_elements,  # 向量的大小
               BLOCK_SIZE: tl.constexpr,  # 每个程序应处理的元素数量
               ):
    """
    Triton 向量相加内核
    每个程序处理 BLOCK_SIZE 个元素
    """
    # 获取当前程序的 PID
    pid = tl.program_id(axis=0)

    # 计算当前程序处理的数据偏移
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建掩码以防止越界访问
    mask = offsets < n_elements

    # 从内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 执行相加操作
    output = x + y

    # 将结果写回内存
    tl.store(output_ptr + offsets, output, mask=mask)


def add_triton_framework(x: torch.Tensor, y: torch.Tensor):
    """
    Triton 向量相加启动函数
    """
    # 预分配输出张量
    output = torch.empty_like(x)
    n_elements = output.numel()

    # 使用 lambda 函数定义网格大小
    def grid(meta): return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # 启动内核
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output