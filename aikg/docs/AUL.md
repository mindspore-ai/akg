# AUL: AI Unity Language

## 简介

AUL (AI Unity Language) 是专门为LLM辅助AI算子生成而设计的一套统一表达语言。其目的是让各类AI模型更简单、明确地设计算子算法、优化调度方案、对接多硬件后端。

## 核心理念

- AUL是Python-like语言，简洁描述算子核心设计，无需严格语法
- AUL供LLM理解，无需实际执行
- AUL语法可根据硬件需求定制
- AUL代码追求简洁，避免冗余
- AUL可灵活扩展新操作

## 重要约束

- **必须** 仅限使用本文档定义的AUL语法
- **必须** 校验代码，避免使用未定义语法、Python、后端语言或第三方库
- **禁止** 使用Python的with语句或其他AUL未定义的语法结构

---

# 第一部分：通用AUL

这部分定义了AUL的基础语法和通用功能，适用于所有硬件平台。

## 基础数据类型

| 类型 | 描述 | 用法示例 |
|-----|-----|---------|
| TensorPtr | Tensor指针 | `input: U.TensorPtr` |
| BufferPtr | 内存指针 | `buffer: U.BufferPtr` |
| Tile | 数据块单元 | `tile = U.Tile(shape=(M, N, K), dtype=U.dtype)` |

## 算子函数签名

```python
# 标准AUL算子函数签名
def operator_name(input1: U.TensorPtr, input2: U.TensorPtr, output: U.TensorPtr):
    # 函数体
```

**参数说明：**
- `input/output`: Tensor数据输入输出首地址

**注意：** 用到的Shape信息按照Tiling函数中的值，直接硬编码写入AUL函数主体

## 基础操作函数

```python
# 创建Tile示例
tile = U.Tile(shape=(M, N), dtype=U.dtype)

# 数据搬移操作示例
U.data_copy(dst=dst_buffer, src=src_buffer)

# 数据填充示例
scalar_127 = U.FilledTile((M, N), U.float32, U.VecBuf, value=127.0)
```

## 基础计算操作

- 算术操作: `+`, `-`, `*`, `/`
- 比较操作: `==`, `!=`, `>`, `<`, `>=`, `<=`
- 索引操作: `tensor[start:end]`, `tile[start:end]`

> **重要：** 硬件后端可扩展计算操作（见AUL-NPU等扩展部分）

---

# 第二部分：AUL-NPU扩展

这部分扩展了AUL，增加了针对神经网络处理器(NPU)的特定功能和优化。

## NPU内存层次

| 层次 | 描述 |
|-----|-----|
| GlobalMem | 全局内存，容量大但访问慢 |
| VecBuf | 向量缓存，用于向量计算的高速缓存 |
| CubeBuf | 矩阵乘内存，用于优化矩阵运算 |

## NPU并行控制

```python
# 获取当前核的唯一标识示例
core_idx = U.get_core_idx()  # 范围为[0, CORE_NUM-1]

# 软件流水线循环 (Software Pipelining)
# U.Pipelined 用于指示循环的迭代可以重叠执行，以隐藏延迟
# iterations 参数指定逻辑迭代的总次数
for iter_idx in U.Pipelined(iterations=LOOP_COUNT):
    # 循环体内的操作
    # 其数据加载和计算预期会由后端进行重叠调度
    pass
```

**核心API说明：**
- `get_core_idx()`: 获取当前核心ID，用于多核并行处理。
- `U.Pipelined(iterations=N)`: 定义N次迭代循环，**强烈暗示**迭代重叠执行。

> **重要：** 关于流水线的详细说明，请参考文档最后的《面向 AUL 的 NPU 架构与流水线概念》部分。

## NPU内存操作

```python
# 创建指定位置Tile示例
tile = U.Tile(shape=(M, N), dtype=U.dtype, pos=U.VecBuf)

# 特定内存位置间数据搬移示例
U.data_copy(dst=dst_buffer, src=src_buffer, src_pos=U.GlobalMem, dst_pos=U.VecBuf)
```

**参数说明：**
- `U.Tile(...)`: 创建固定切分块的Tensor。
- `U.data_copy(...)`: 在不同内存空间拷贝数据。

## NPU Tile级别计算操作

AUL采用通用向量接口，用`op`参数指定操作，灵活扩展：

### 向量二元操作

```python
# 向量二元操作示例
U.vbinary_op(op="add", dst=dst_tile, src1=tile_a, src2=tile_b) # 加法操作
U.vbinary_op(op="mul", dst=dst_tile, src1=tile_a, src2=tile_b) # 乘法操作

# 支持的操作类型包括：add, sub, mul, div 等
# 扩展二元操作示例
U.vbinary_op(op="xxx", dst=dst_tile, src1=tile_a, src2=tile_b)
```

### 向量一元操作

```python
# 向量一元操作示例
U.vunary_op(op="sqrt", dst=dst_tile, src=src_tile) # 平方根
U.vunary_op(op="exp", dst=dst_tile, src=src_tile)  # 指数
U.vunary_op(op="log", dst=dst_tile, src=src_tile)  # 对数
U.vunary_op(op="relu", dst=dst_tile, src=src_tile) # ReLU
U.vunary_op(op="abs", dst=dst_tile, src=src_tile)  # 绝对值

# 扩展一元操作示例
U.vunary_op(op="xxx", dst=dst_tile, src=src_tile)
```

### 向量规约操作

```python
# 向量规约操作示例
U.vreduce_op(op="sum", dst=dst_tile, src=src_tile, axis=-1) # 求和规约
U.vreduce_op(op="max", dst=dst_tile, src=src_tile, axis=0) # 最大值规约

# 注意: 
# 1. mean操作需要先sum再div
```

### 矩阵操作

```python
# 矩阵乘法操作示例
U.matmul_op(dst=dst_tile, src1=a_tile, src2=b_tile)

# 注意: src1、src2数据类型必须一致
```

### 向量标量操作

```python
# 向量与标量操作示例
U.vectorscalar_op(op="adds", dst=dst_tile, src=src_tile, factor=3.14) # 加标量
U.vectorscalar_op(op="muls", dst=dst_tile, src=src_tile, factor=2.0)  # 乘标量
```

---

# AUL语法速查表

## 通用AUL速查

| 类别 | 语法/API | 描述 |
|------|----------|------|
| **类型** | `U.TensorPtr`, `U.BufferPtr`, `U.Tile` | 基础数据类型 |
| **数据操作** | `U.Tile(shape, dtype)` | 创建数据块 |
|  | `U.data_copy(dst, src)` | 数据搬移 |
|  | `U.FilledTile(shape, dtype, value)` | 数据填充 |
| **计算操作** | `+`, `-`, `*`, `/` | 基础算术操作 |
|  | `==`, `!=`, `>`, `<`, `>=`, `<=` | 比较操作 |
| **索引操作** | `[]` | 数据索引 |


## 扩展AUL-NPU速查

| 类别 | 语法/API | 描述 |
|------|----------|------|
| **内存位置** | `GlobalMem`, `VecBuf`, `CubeBuf` | NPU内存层次 |
| **硬件控制** | `U.get_core_idx()` | 获取核心ID |
|  | `U.Pipelined(iterations=N)` | 流水线循环 |
| **扩展数据操作** | `U.Tile(shape, dtype, pos)` | 扩展Tile创建 |
|  | `U.data_copy(dst, src, src_pos, dst_pos)` | 扩展数据搬移 |
| **向量二元操作** | `U.vbinary_op(op="add\|mul\|div\|sub\|...", dst, src1, src2)` | 向量二元运算 |
| **向量一元操作** | `U.vunary_op(op="sqrt\|exp\|log\|relu\|abs\|...", dst, src)` | 向量一元运算 |
| **向量规约操作** | `U.vreduce_op(op="sum\|max\|min\|...", dst, src, axis=-1)` | 向量规约运算 |
| **向量矩阵操作** | `U.matmul_op(dst, src1, src2)` | 矩阵乘法 |
| **向量标量操作** | `U.vectorscalar_op(op="adds\|muls\|maxs\|mins\|...", dst, src, factor)` | 向量标量运算 |
| **标量标量操作** | `+`, `-`, `*`, `/`, `==`, `!=`, `>`, `<`, `>=`, `<=` | 标量间基础运算 |


## AUL编程规范

1. **先设计算法**：确定通用AUL逻辑
2. **再硬件优化**：添加目标硬件AUL优化
3. **保持简洁**：避免冗余代码
4. **正确用内存**：明确数据移动路径
5. **活用操作函数**：用`op`参数指定操作

**特别注意：** AUL为LLM设计，是描述性语言而非执行语言。目的是简化算子描述，减少平台差异，保留优化能力。

---

# 总结

AUL作为一种统一的AI算子表达语言，其主要价值在于：

1. **简化描述**：简洁直观描述算子
2. **跨平台**：减少硬件差异
3. **保留优化**：允许硬件特定优化
4. **LLM友好**：为LLM设计，非严格语法

AUL助LLM在不同硬件上实现一致开发体验，兼顾性能与灵活性。

---

# 面向 AUL 的 NPU 架构与流水线概念

## NPU 硬件概述

本节介绍 NPU 的典型硬件构成，作为理解 AUL 中 NPU 相关特性 (如内存位置、流水线) 的背景知识。

### 整体结构与核心组件

-   **全局内存 (GlobalMem):** 整个 NPU 芯片共享的大容量、相对低速的内存（例如，容量 40GB）。所有持久化的 Tensor 数据通常存储在这里。
-   **NPU 核心 (NPU Core):** 芯片包含多个（例如 8 个）物理处理核心，是执行计算的主要单元。代码通常会在这些核心上并行执行。

### NPU 核心内部资源

每个 NPU 核心通常包含以下关键部分：

-   **向量缓存 (VecBuf):** 核心专用的高速、小容量片上内存 (例如 256KB)。这是计算单元直接操作数据的地方，计算前需要将数据从 GlobalMem 加载到这里。对数据存放地址可能有对齐要求 (例如 256Bytes)。

-   **执行单元 (Execution Units):** 并行操作的专用硬件单元：
    *   **加载单元 (Load Unit):** 负责将数据从 GlobalMem 传输 *到* 核心的向量缓存 (VecBuf)。
    *   **存储单元 (Store Unit):** 负责将数据 *从* 向量缓存 (VecBuf) 传输回 *到* GlobalMem。
    *   **向量计算单元 (Vector Compute Unit):** 对驻留在向量缓存 (VecBuf) 中的数据执行实际计算 (如向量加法、乘法、规约)。可能仅支持连续或带掩码 (masked) 的访问模式。
    *   **标量计算单元:** 类似一个小型 CPU，可以执行一些标量运算，用于控制流或辅助计算。

### 执行模型与并行性

-   **数据流路径:** 典型数据路径是：GlobalMem → (加载单元) → VecBuf → (计算单元) → VecBuf → (存储单元) → GlobalMem。
-   **核心并行性:** NPU 设计的核心思想是并行。不同的 NPU 核心可以并行工作。
-   **单元并行性:** 核心内部的加载、存储和计算单元也 **可以并发操作**。例如，加载单元可以在计算单元处理当前数据块的同时，取回下一个数据块。
-   **数据依赖与同步:** 对于仅使用 *一组缓冲区* 处理的数据，存在严格的串行依赖：必须先完成 `Load`，该数据的 `Compute` 才能开始；必须先完成 `Compute`，`Store` 才能写回结果。
-   **同步机制:** 底层通常通过类似 `set_flag / wait_flag` 机制实现同步。如果管理不当，会导致单元空闲等待 (stalls)。

## AUL 流水线表达

AUL 提供了高级抽象，简化了流水线编程：

-   **`U.Pipelined(iterations=N)`:** 高级指令，告知 LLM/编译器，循环迭代期望以流水线方式执行，利用硬件的并行能力。

## AUL-NPU 流水线示例：向量加法

```python
import aul as U

def vector_add_pipelined(A: U.TensorPtr, B: U.TensorPtr, C: U.TensorPtr)
    
    # 1. 解析配置参数
    TILE_LEN = 256
    LOOP_COUNT = 5
    total_len = 10240
    CORE_NUM = 8
    
    # 2. 获取核心ID并计算数据范围
    core_idx = U.get_core_idx()
    len_per_core = total_len // CORE_NUM
    start_idx = core_idx * len_per_core
    end_idx = start_idx + len_per_core
    
    # 3. 使用Pipelined循环实现流水线
    for i in U.Pipelined(iterations=LOOP_COUNT):
        # 3.1 计算当前迭代数据索引
        current_start = start_idx + i * TILE_LEN
        current_end = current_start + TILE_LEN
        
        # 3.2 创建Tile
        a_tile = U.Tile(shape=(TILE_LEN,), dtype=A.dtype, pos=U.VecBuf)
        b_tile = U.Tile(shape=(TILE_LEN,), dtype=B.dtype, pos=U.VecBuf)
        c_tile = U.Tile(shape=(TILE_LEN,), dtype=C.dtype, pos=U.VecBuf)
        
        # 3.3 流水线阶段1: 加载输入数据
        U.data_copy(dst=a_tile, src=A[current_start:current_end],
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        U.data_copy(dst=b_tile, src=B[current_start:current_end],
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 3.4 流水线阶段2: 执行计算
        U.vbinary_op(op="add", dst=c_tile, src1=a_tile, src2=b_tile)
        
        # 3.5 流水线阶段3: 写回结果
        U.data_copy(dst=C[current_start:current_end], src=c_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)
```

---

# 实用指南：AUL代码生成最佳实践

## 转换步骤

1. **理解任务**：完全理解算子执行的功能逻辑
2. **设计基础算法**：使用通用AUL设计基础算法流程
3. **添加硬件优化**：基于目标硬件特性添加优化，例如流水线
4. **验证代码**：检查是否符合AUL语法规范

## 常见错误避免

1. **使用未定义API**：仅使用文档中定义的API，不要"发明"新API
2. **忽略内存层次**：明确数据在不同内存层次间的移动
3. **错误的优化时机**：先保证正确性，再进行优化
4. **不恰当的批处理**：考虑算子输入大小，合理划分批处理大小
5. **尝试实现双缓存机制**：双缓存会在后续编译器流程实现，aul代码设计不需要考虑这部分
6. **不同大小的张量进行运算**：计算前根据预先申请的Tile，核验Tensor的大小
7. **涉及reduce操作的维度进行切分**：注意切分循环的维度，尽量不要切分涉及reduce操作的轴
8. **广播机制**：AUL不支持广播机制，需要注意操作的张量的大小
9. **切分方法**：如果数据大小超过存储限制，需要再次切分，保证切分与核间切分是同一个轴，例如核间并行切了第二个轴，for循环也切第二个轴
10. **不支持语法**：AUL不支持.shape操作，请显式的写出shape

## 检查清单

- [ ] 函数签名正确，包含必要参数
- [ ] 内存操作明确指定源和目标位置
- [ ] 优化使用适当，如多核并行
- [ ] 没有使用未定义的操作或语法
- [ ] 代码结构清晰，逻辑正确
