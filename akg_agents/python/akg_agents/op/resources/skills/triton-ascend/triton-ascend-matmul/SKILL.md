---
name: triton-ascend-matmul
description: "矩阵乘法算子(matmul)优化策略和切分技巧"
level: L4
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  operator_patterns: "matmul"
  algorithms: "matmul, bmm, linear"
---

# MatMul 算子优化

> 适用于矩阵乘法及相关运算

## Ascend 后端切分优化

**关键原则**: 充分发挥带宽，算子行宽为 512B 的整数倍。

以 fp16/bf16 为例（每个元素 2 字节）：

### 切分配置（根据转置情况）

1. **A、B 都不转置**
   - 分块行宽分别为 K0 和 N0
   - **推荐**: M0=128, K0=256, N0=256

2. **A 不转置，B 转置**
   - 分块行宽都是 K0
   - **推荐**: K0=256, M0 和 N0 影响较小

3. **A、B 都转置**
   - 分块行宽分别为 M0 和 K0
   - **推荐**: M0=256, K0=256, N0=128

4. **A 转置，B 不转置**
   - 分块行宽分别为 M0 和 N0
   - **注意**: 左右矩阵均无法同时满足 512B 的整数倍，需根据实际情况调整

### 为什么是 512B？

- 512B = 256 个 fp16/bf16 元素（256 × 2 字节）
- NPU 的最佳带宽对齐
- 确保每次内存访问充分利用带宽

## 固定核心数启动

MatMul 算子使用 **CUBE核心数**（矩阵计算核心）。

**关键**: 使用 `grid=(num_cores,)` 而非 `(NUM_BLOCKS,)`

```python
import torch_npu

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在__init__中获取 CUBE 核心数
        try:
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except:
            self.CUBE_CORE_NUM = 20  # Ascend 910B4 默认
    
    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256
        
        # 关键：固定核心数启动
        matmul_kernel[(self.CUBE_CORE_NUM,)](
            a, b, c, M, N, K, self.CUBE_CORE_NUM,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
        return c
```

## 标准 MatMul Kernel

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 核心 ID: 0 ~ num_cores-1
    pid = tl.program_id(0)
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 计算当前块的 2D 索引
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N
        
        # 初始化累加器（使用 float32 提高精度）
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K 维度循环
        for k in range(0, K, BLOCK_K):
            # 加载 A 块
            a_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * K + \
                       (k + tl.arange(0, BLOCK_K))[None, :]
            a_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M
            a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)
            
            # 加载 B 块
            b_offset = (k + tl.arange(0, BLOCK_K))[:, None] * N + \
                       (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
            b_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N
            b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)
            
            # 矩阵乘累加
            accumulator += tl.dot(a, b)
        
        # 存储结果
        c_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * N + \
                   (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        c_mask = ((block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M) & \
                 ((block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N)
        tl.store(c_ptr + c_offset, accumulator, mask=c_mask)
```

## 使用 block_ptr 优化（推荐）

对于 2D 矩阵，使用 `tl.make_block_ptr` 可以简化代码并提高性能：

```python
@triton.jit
def matmul_kernel_optimized(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N
        
        # 创建 block pointers
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(block_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0)
        )
        
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, block_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0)
        )
        
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K 维度循环
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))
            accumulator += tl.dot(a, b)
            
            # 移动 block pointers
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
        
        # 存储结果
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(block_m * BLOCK_M, block_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))
```

## 优化要点

### 1. 核心数配置

- ✅ 使用 CUBE_CORE_NUM（矩阵计算核心）
- ✅ 在 `__init__` 中获取，避免 forward 中重复调用
- ✅ `grid=(num_cores,)` 固定启动

### 2. 切分配置

- ✅ 行宽对齐到 512B（256 个 fp16 元素）
- ✅ 根据转置情况选择合适的 M0/N0/K0
- ✅ 常用配置: (128, 256, 256)

### 3. 精度控制

- ✅ 累加器使用 float32: `tl.zeros(..., dtype=tl.float32)`
- ✅ 即使输入是 fp16/bf16，也用 float32 累加
- ✅ 最后存储时自动转回目标精度

### 4. 内存访问

- ✅ 优先使用 `tl.make_block_ptr` 和 `boundary_check`
- ✅ 注意 stride 参数设置
- ✅ 使用 `tl.advance` 移动块指针

## 性能检查清单

- [ ] 是否使用了 CUBE 核心数？
- [ ] 切分配置是否对齐到 512B？
- [ ] 累加器是否使用 float32？
- [ ] 是否使用固定核心数启动（而非总块数）？
- [ ] K 维度循环是否正确实现？

## 常见错误

1. **使用 VEC 而非 CUBE**: MatMul 应该用 CUBE 核心
2. **直接用总块数启动**: 应该用 `grid=(num_cores,)` + 循环
3. **切分未对齐 512B**: 性能不佳
4. **累加用 fp16**: 精度损失严重
5. **忘记 K 维度循环**: 结果错误
