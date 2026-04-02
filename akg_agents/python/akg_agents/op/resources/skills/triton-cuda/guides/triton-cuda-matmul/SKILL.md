---
name: triton-cuda-matmul
description: "矩阵乘法算子(matmul/bmm/linear)优化策略，包括分块 Tiling、共享内存缓存、Tensor Core 利用和大矩阵处理技巧。适用于实现 GEMM、批量矩阵乘、全连接层等矩阵运算的 CUDA 内核代码生成场景"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "matmul"
  algorithms: "matmul, bmm, linear"
---

# MatMul 算子优化

> 适用于矩阵乘法及相关运算

## CUDA GPU MatMul 优化核心

### Tensor Core 利用

- **Ampere (A100)**: 支持 FP16, BF16, TF32, INT8 Tensor Core
- **Hopper (H100)**: 额外支持 FP8, wgmma 指令
- **关键**: `tl.dot(a, b, allow_tf32=True)` 启用 TF32 Tensor Core

### 分块配置建议

常用配置（2 的幂次）：

| 配置 | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages | 适用场景 |
|------|---------|---------|---------|-----------|------------|---------|
| 小矩阵 | 64 | 64 | 32 | 4 | 4 | M, N < 1024 |
| 中矩阵 | 128 | 128 | 32 | 4 | 3 | M, N < 4096 |
| 大矩阵 | 128 | 256 | 64 | 8 | 3 | M, N >= 4096 |
| 高 K | 64 | 128 | 64 | 4 | 4 | K 很大 |

## 标准 MatMul Kernel（使用 block_ptr）

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # 2D 索引计算
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 创建 block pointers
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    # 使用 float32 累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # K 维度循环
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        
        # 移动 block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    
    # 存储结果（需显式转换类型，匹配输出 dtype）
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
```

## 使用 Autotune 优化

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
    restore_value=['c_ptr'],  # 必须：列出所有输出指针参数名
)
@triton.jit
def matmul_kernel_autotune(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2 缓存优化：Grouped ordering
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # ... 后续与标准 kernel 相同
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
```

## Host 侧启动

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        
        matmul_kernel_autotune[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c
```

## L2 缓存优化：Grouped Ordering

### 为什么需要 Grouped Ordering？

标准的行优先或列优先遍历会导致 L2 缓存利用率低。通过将相邻的块分组处理，可以增加数据复用：

```python
# 标准遍历：相邻 pid 访问不同行的 A 块
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

# Grouped ordering：相邻 pid 访问同一组行的 A 块
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### Swizzle2D

另一种缓存优化方式：
```python
task_m, task_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)
```

## 优化要点

### 1. 分块配置

- 使用 autotune 搜索最优配置
- 考虑 Tensor Core 的要求（块大小为 16 的倍数）
- 更大的块 → 更好的数据复用，但更高的寄存器压力

### 2. 精度控制

- 累加器使用 float32: `tl.zeros(..., dtype=tl.float32)`
- 即使输入是 fp16/bf16，也用 float32 累加
- 最后存储时自动转回目标精度

### 3. 内存访问

- 优先使用 `tl.make_block_ptr` 和 `boundary_check`
- 使用 `tl.advance` 移动块指针
- 利用 Grouped Ordering 优化 L2 缓存

### 4. 流水线

- `num_stages` 控制软件流水线级数
- 更多 stage → 更好地隐藏内存延迟
- 但会占用更多共享内存

## 性能检查清单

- [ ] 是否使用了 autotune 搜索最优配置？
- [ ] 累加器是否使用 float32？
- [ ] 是否使用了 Grouped Ordering 或 swizzle2d 优化 L2 缓存？
- [ ] K 维度循环是否正确实现？
- [ ] num_warps 和 num_stages 是否合理？
- [ ] block 大小是否为 16 的倍数（Tensor Core 要求）？

## 常见错误

1. **累加用 fp16**: 精度损失严重
2. **忘记 K 维度循环**: 结果错误
3. **block 大小不对齐 Tensor Core**: 性能不佳
4. **L2 缓存未优化**: 大矩阵性能下降
5. **num_warps 不匹配**: block 大小和 warp 数不匹配导致资源浪费
