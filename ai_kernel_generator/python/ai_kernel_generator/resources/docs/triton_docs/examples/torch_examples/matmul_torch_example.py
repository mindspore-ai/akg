import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(output_ptr, x_ptr, y_ptr,
                  A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    """
    Triton 矩阵乘法内核
    计算 X @ Y，其中 X 是 (B, C)，Y 是 (C, D)
    """
    # 创建索引
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    # 计算数据索引
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]

    # 加载数据
    X = tl.load(x_ptr + Xidx)
    Y = tl.load(y_ptr + Yidx)

    # 执行矩阵乘法
    result = tl.dot(X, Y)

    # 计算输出索引并存储结果
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, result)


def matmul_triton_framework(x0, x1):
    """
    Triton 矩阵乘法启动函数
    """
    B, C = x0.shape
    C2, D = x1.shape
    assert C == C2, f"矩阵维度不匹配: {C} != {C2}"

    # 分配输出张量
    output = torch.zeros((B, D), dtype=x0.dtype, device=x0.device)

    # 启动内核
    matmul_kernel[1, 1, 1](output, x0, x1, 1, B, C, D)

    return output
