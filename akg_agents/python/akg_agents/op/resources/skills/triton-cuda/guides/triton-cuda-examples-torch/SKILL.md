---
name: triton-cuda-examples-torch
description: "PyTorch 框架下 Triton CUDA 内核的完整集成示例，包括 vector_add、matmul、layer_norm、softmax 等标准算子实现。适用于需要参考 PyTorch 算子包装方式、torch.autograd.Function 实现模式的 CUDA 内核代码生成场景"
category: example
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  framework: torch
  examples: "vector_add, matmul, layer_norm, softmax, double_kernel"
---

# PyTorch + Triton CUDA 示例代码

本 Skill 包含完整的可运行示例代码，展示如何在 PyTorch 中使用 Triton CUDA 编写高性能 kernel。

## 示例列表

### 1. Vector Add（向量加法）
**算子类型**: Element-wise
**关键点**:
- 最简单的 Triton kernel 示例
- 一维索引和 mask
- 标准五步模式

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton 向量相加内核，每个程序处理 BLOCK_SIZE 个元素"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
```

### 2. Softmax
**算子类型**: Reduce
**关键点**:
- 数值稳定化（减去最大值）
- 逐行处理，grid stride loop
- `tl.range` 和 `tl.num_programs` 配合使用

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # 数值稳定性
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        y = torch.empty_like(x)
        num_programs = min(32, n_rows)
        softmax_kernel[(num_programs, 1, 1)](
            y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE
        )
        return y
```

### 3. Layer Norm（层归一化）
**算子类型**: Reduce + Element-wise
**关键点**:
- 多遍扫描（均值→方差→归一化）
- float32 中间计算
- 保存统计量供反向传播

```python
import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # 第一遍：计算均值
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

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # 第三遍：归一化
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, normalized_shape, weight, bias, eps=1e-5):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = 1024
        layer_norm_kernel[(M,)](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return y
```

### 4. MatMul（矩阵乘法）
**算子类型**: MatMul
**关键点**:
- 使用 `tl.dot` 进行矩阵乘法
- 2D 索引计算
- block_ptr 简化访问

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        )
        return c
```

### 5. Double Kernel（双内核调用）
**算子类型**: 多 Kernel 组合
**关键点**:
- 在一个 forward 中调用多个 kernel
- 中间结果通过 tensor 传递
- 每个 kernel 独立配置 grid 和 block

```python
import torch
import triton
import triton.language as tl

@triton.jit
def first_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(data, 0.0)  # ReLU
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def second_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    result = data * data  # Square
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # 第一个 kernel: ReLU
        intermediate = torch.empty_like(x)
        first_kernel[grid](x, intermediate, n_elements, BLOCK_SIZE)

        # 第二个 kernel: Square
        output = torch.empty_like(x)
        second_kernel[grid](intermediate, output, n_elements, BLOCK_SIZE)

        return output
```

## 通用模式

所有示例都遵循相同的结构：

### Kernel 定义
```python
@triton.jit
def kernel_name(
    output_ptr, input_ptr,   # 输入/输出指针
    M, N, K,                  # 形状参数
    BLOCK_SIZE: tl.constexpr, # 编译时常量
):
    pid = tl.program_id(0)
    offsets = ...
    mask = ...
    data = tl.load(...)
    result = compute(data)
    tl.store(...)
```

### ModelNew 类
```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *inputs):
        M, N = inputs[0].shape
        output = torch.empty_like(inputs[0])
        grid = (triton.cdiv(M, BLOCK_SIZE),)
        kernel_name[grid](output, inputs[0], M, N, BLOCK_SIZE=1024)
        return output
```

## 关键注意事项

### 1. 张量设备和数据类型
```python
# 确保输出张量与输入在同一设备
output = torch.empty_like(input_tensor)  # 推荐
# 或
output = torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)
```

### 2. Grid 配置
```python
# 简单情况：直接计算
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

# 2D 情况
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# autotune 情况：使用 lambda
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)

# 限制程序数（逐行处理）
grid = (min(n_rows, 32),)
```

### 3. ModelNew 格式要求
- **必须**继承 `torch.nn.Module`
- **必须**实现 `forward` 方法
- 输出张量使用 `torch.empty_like` 或 `torch.empty`

### 4. 参数传递
```python
# 正确：所有参数作为位置参数传递，constexpr 使用关键字
kernel[grid](output, input, M, N, BLOCK_SIZE=1024)

# 错误：非 constexpr 参数使用关键字
kernel[grid](output=output, input=input)
```

## 验证正确性
```python
# 与 PyTorch 原生实现对比
x = torch.randn(128, 256, device='cuda', dtype=torch.float16)
output_triton = model_new(x)
output_torch = torch.nn.functional.softmax(x, dim=-1)

# 检查差异
diff = (output_triton - output_torch).abs().max()
print(f"Max difference: {diff.item()}")
assert diff < 1e-3, "Results mismatch!"
```
