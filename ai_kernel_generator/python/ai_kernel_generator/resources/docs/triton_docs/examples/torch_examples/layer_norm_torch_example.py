import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X,  # 输入指针
    Y,  # 输出指针
    W,  # 权重指针
    B,  # 偏差指针
    Mean,  # 均值指针
    Rstd,  # 1/std 指针
    stride,  # 行步长
    N,  # 特征维度
    eps,  # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton Layer Normalization 内核
    每个程序处理一行数据
    """
    # 获取当前程序处理的行
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # 第一遍：计算均值
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # 第二遍：计算方差
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # 保存 mean 和 rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # 第三遍：归一化并应用线性变换
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


@torch.inference_mode()
def layer_norm_triton_framework(x, normalized_shape, weight, bias, eps=1e-5):
    """
    Triton Layer Normalization 启动函数
    """
    # 分配输出张量
    y = torch.empty_like(x)

    # 将输入展平为二维
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    # 分配中间结果张量
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = 1024

    # 启动内核
    layer_norm_kernel[(M, )](
        x_arg, y, weight, bias, mean, rstd,
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return y
