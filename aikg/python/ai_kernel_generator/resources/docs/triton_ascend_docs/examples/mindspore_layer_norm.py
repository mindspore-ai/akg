import torch
import triton
import triton.language as tl
import mindspore as ms


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


class ModelNew(ms.nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def construct(self, x, weight, bias):
    """
        Triton Layer Normalization
    """
    # 分配输出张量
    y = ms.mint.empty_like(x)

    # 将输入展平为二维
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    # 分配中间结果张量
    mean = ms.mint.empty((M, ), dtype=ms.float32)
    rstd = ms.mint.empty((M, ), dtype=ms.float32)

    BLOCK_SIZE = 1024

    # 启动内核
    layer_norm_kernel[(M, )](
        x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0), N, self.eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return y

# if __name__ == "__main__":
#     M = 128
#     N = 128
#     x_shape = (M, N)
#     w_shape = (x_shape[-1], )
#     weight = ms.mint.rand(w_shape, dtype=ms.float16)
#     bias = ms.mint.rand(w_shape, dtype=ms.float16)
#     x = -2.3 + 0.5 * ms.mint.randn(x_shape, dtype=ms.float16)
#     dy = .1 * ms.mint.randn_like(x)
#     # 前向传播
#     y_tri = layer_norm_triton_mindspore(x, w_shape, weight, bias, 1e-5)
#     print(y_tri)
