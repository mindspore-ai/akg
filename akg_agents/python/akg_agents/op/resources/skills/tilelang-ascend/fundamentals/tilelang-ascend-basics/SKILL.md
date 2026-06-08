---
name: tilelang-ascend-basics
description: "TileLang Ascend 编码基础规范：代码模板、函数设计原则、维度参数自推导、融合算子 workspace 配置模式、Host 预处理声明。所有算子生成任务必须遵守的基础编码约定。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang Ascend 编码基础规范

---

## 标准内核结构和启动函数模板

### 代码模板

TileLang Ascend 算子由三层嵌套构成：

```python
@tilelang.jit(out_idx=[...], pass_configs=...)
def kernel(编译期参数):
    @T.prim_func
    def main(运行期tensor参数):
        ...  # 计算逻辑
    return main

class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()

    def forward(self, *inputs):
        kernel_func = kernel(编译期参数)  # 第1次调用：传入编译期参数，返回可执行函数
        outputs = kernel_func(*inputs)  # 第2次调用：传入tensor
        return outputs
```

### `@jit(out_idx=[...])` 调用约定

- **传入 out_idx**：自动分配输出 tensor 并**返回**，不会原地写入

```python
outputs = kernel_func(*inputs)
```

- **不传入 out_idx**：从 host 侧传入输出 tensor，原地写入

```python
output_i = torch.empty/empty_like/zeros/zeros_like()
kernel_func(*inputs, *outputs)
```

### 尺寸变量应定义在 @jit 层，而非 @T.prim_func 内部

`@jit` 层的变量在编译时已被具体值替换，`@T.prim_func` 内的变量会被 TVM 保留为符号表达式，导部分后续计算报错。

```python
@jit(...)
def kernel(M, N, block_M, block_N, dtype):
    # 正确做法：在 @jit 层定义，令TVM 推到具体值
    sub_block_M = block_M // 2
    m_num = M // block_M

    @T.prim_func
    def main(A, C):
        # 错误做法：在 @T.prim_func 内部定义
        # sub_block_M = block_M // 2

        buf = T.alloc_shared((sub_block_M, block_N), dtype)
        T.tile.mul(buf, buf, -1.0)
```

## 算子设计

### 核心决策
- **编程模式选型**：Developer / Expert / 混合模式
- **API 映射**：将数学公式拆解为 TileLang DSL 原语组合
- **内存层级规划**：GM → L1/UB → L0 的数据搬运路径
- **Tiling 策略**：Block 划分与 Tile Shape 设计
- **循环结构**：T.Parallel / T.serial / T.Pipelined / T.Persistent 的选择
- **同步策略**：自动同步 vs 手动同步标志

### 已知限制

| 约束 | 说明 | 影响 | 替代方案 |
|------|------|------|----------|
| **不支持三维 Kernel** | `T.Kernel` 只接受一维 block 数 | 三维并行设计无法实现 | 使用 `block_metadata` 预计算机制 |
| **threads 参数限制** | 只支持 1 或 2，不支持大值 | `threads=128` 等设计报错 | 默认不指定 threads 或设为 2 |
| **动态循环边界不支持** | 循环次数不能依赖 tensor 值（如 `batch_sizes[bz]`） | `T.Pipelined(batch_sizes[bz])` 报错 | 预计算最大循环次数，用 `T.serial(max_iters)` + 条件判断 |
| **流水线不支持动态边界** | `T.Pipelined` 的循环次数必须静态 | 动态批次无法流水线 | 改用 `T.serial` 或预计算固定迭代次数 |
| **部分 GPU API 不可用** | CUDA 专用 API 在 Ascend 不存在 | 直接移植 GPU 代码失败 | 查阅 API 章节确认 Ascend API |
| **GEMM 要求 M,N 为 block 整数倍** | `M // block_M` 整除依赖；`M < block_M` 时零 block 启动 | 输出全零或除零编译崩溃 | 必须明确处理策略：host 侧 padding+crop 或 Kernel 动态 block |
| **L0C 容量上限** | A2/A3 设备 L0C = 128KB | `block_M × block_N × sizeof(accum) > 128KB` 导致 segfault | 设计 block 时满足 `block_M × block_N ≤ 16384`（float32 accum） |

## 关键编码规范

### GEMM 算子：非整除维度处理

GEMM kernel 内部使用 `M // block_M` 和 `N // block_N`，要求 M、N 为 block 大小整数倍。非整除时需在调用的 Python 层 zero-padding 后裁剪：

```python
# padding
M_pad = ((M + block_M - 1) // block_M) * block_M
N_pad = ((N + block_N - 1) // block_N) * block_N
K_pad = ((K + block_K - 1) // block_K) * block_K

if M_pad > M or K_pad > K:
    kernel_padded = torch.zeros(M_pad, K_pad, ...)
    kernel_padded[:M, :K] = kernel_flat

# GEMM 后裁剪
output = output[:M, :N]
```

**关键约束**: 不 padding 时 `M // block_M = 0`（当 M < block_M）会导致零 block 启动（输出全零）或除零编译崩溃。

### Autotune 算子: supply_prog 与 get_configs 接口约定

- **`supply_prog(params)`**: `params` 仅含输入 tensor 描述符（不含输出 param）。从 `params[0].shape` / `params[1].shape` 提取维度，不可访问 `params[2]`。
- **`get_configs` 作为 callable**: autotuner 调用形式为 `get_configs(key_args_tuple, key_kwargs_tuple)`，须签名为 `get_configs(key_args, _key_kwargs=None)`，从 `key_args` 提取 M/N/K。
- **config 过滤**: 必须在 `get_configs` 中过滤 `block > dimension` 的无效组合（避免除零编译错误），及 `block_M * block_N * sizeof(accum) > L0C_capacity` 的组合（避免 L0C 溢出 segfault）。

### Buffer 分配

```python
# VEC_NUM = 2，每个 vector 核处理 block_M // VEC_NUM 行
a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
```

### 数据搬运索引

```python
# 标准索引模式
row_start = bx * block_M + vid * block_M // VEC_NUM
T.copy(A[row_start, by * block_N], a_ub)
T.copy(a_ub, B[row_start, by * block_N])
```

### 同步

```python
# Expert 模式：手动同步
with T.Scope("V"):
    T.copy(A[...], a_ub)
    T.barrier_all()
    T.tile.exp(a_ub, a_ub)
    T.barrier_all()
    T.copy(a_ub, B[...])

# Developer 模式 + 自动同步：无需手动 barrier
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

### 广播

```python
# 归约结果 [M, 1] 广播到 [M, N]
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
T.tile.broadcast(max_2d_ub, max_ub)
```
