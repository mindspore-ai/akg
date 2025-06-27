# 第三部分：NPU上的AUL拓展

## NPU内存层次

| 层次 | 描述 |
|-----|-----|
| GlobalMem | 全局内存，容量大但访问慢 |
| VecBuf | 向量缓存，用于向量计算的高速缓存 |
| CubeBuf | 矩阵乘内存，用于优化矩阵运算 |


## NPU并行控制

```python
# 获取当前核的唯一标识，范围为[0, CORE_NUM-1]
core_idx = U.get_core_idx()

# 软件流水线循环 (Software Pipelining)
# U.Pipelined 用于指示循环的迭代可以重叠执行，以隐藏延迟
# iterations 参数指定逻辑迭代的总次数
for iter_idx in U.Pipelined(iterations=LOOP_COUNT):
    # 其数据加载和计算预期会由后端进行重叠调度
    pass
```

**核心API说明：**
- `U.get_core_idx()`: 获取当前核心ID，用于多核并行处理。
- `U.Pipelined(iterations=N)`: 定义N次迭代循环，暗示迭代重叠执行。

> **重要：** 关于流水线的详细说明，请参考《面向 AUL 的 NPU 架构与流水线概念》部分。

## NPU内存操作

```python
# 创建指定位置Tile示例
tile = U.Tile(shape=(M, N), dtype=U.dtype, pos=U.VecBuf)

# 特定内存位置间数据搬移
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
U.vunary_op(op="ln", dst=dst_tile, src=src_tile)  # 对数
U.vunary_op(op="relu", dst=dst_tile, src=src_tile) # ReLU
U.vunary_op(op="abs", dst=dst_tile, src=src_tile)  # 绝对值
U.vunary_op(op="cast_fp16_to_fp32", dst=dst_tile, src=src_tile)  # 类型转换
U.vunary_op(op="reshape", dst=dst_tile, src=src_tile, newshape=new_shape)  # 更改数组形状
U.vunary_op(op="vector_dup", dst=dst_tile, fill_shape=fill_shape, fill_value=0.0)  # 填充值

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

## NPU扩展API速查表

| 类别 | 语法/API | 描述 |
|------|----------|------|
| **内存位置** | `GlobalMem`, `VecBuf`, `CubeBuf` | NPU内存层次 |
| **硬件控制** | `U.get_core_idx()` | 获取核心ID |
|  | `U.Pipelined(iterations=N)` | 流水线循环 |
| **扩展数据操作** | `U.Tile(shape, dtype, pos)` | 扩展Tile创建 |
|  | `U.data_copy(dst, src, src_pos, dst_pos)` | 扩展数据搬移 |
| **向量二元操作** | `U.vbinary_op(op="add\|mul\|div\|sub\|...", dst, src1, src2)` | 向量二元运算 |
| **向量一元操作** | `U.vunary_op(op="sqrt\|exp\|ln\|relu\|abs\|reshape\|...", dst, src)` | 向量一元运算 |
| **向量规约操作** | `U.vreduce_op(op="sum\|max\|min\|...", dst, src, axis=-1)` | 向量规约运算 |
| **向量矩阵操作** | `U.matmul_op(dst, src1, src2)` | 矩阵乘法 |
| **向量标量操作** | `U.vectorscalar_op(op="adds\|muls\|maxs\|mins\|...", dst, src, factor)` | 向量标量运算 |
| **标量标量操作** | `+`, `-`, `*`, `/`, `==`, `!=`, `>`, `<`, `>=`, `<=` | 标量间基础运算 |

---

## 面向AUL的NPU架构与流水线概念

### NPU硬件概述

本节介绍 NPU 的典型硬件构成，作为理解 AUL 中 NPU 相关特性 (如内存位置、流水线) 的背景知识。

### 整体结构与核心组件

- **全局内存 (GlobalMem):** 整个 NPU 芯片共享的大容量、相对低速的内存（例如，容量 40GB）。所有持久化的 Tensor 数据通常存储在这里。
- **NPU核心 (NPU Core):** 芯片包含多个物理处理核心，是执行计算的主要单元。代码通常会在这些核心上并行执行。

### NPU核心内部资源

每个 NPU 核心通常包含以下关键部分：

- **向量缓存 (VecBuf):** 核心专用的高速、小容量片上内存 (例如 256KB)。这是计算单元直接操作数据的地方，计算前需要将数据从 GlobalMem 加载到这里。对数据存放地址可能有对齐要求 (例如 256Bytes)。

- **执行单元 (Execution Units):** 并行操作的专用硬件单元：
    * **加载单元 (Load Unit):** 负责将数据从 GlobalMem 传输 *到* 核心的向量缓存 (VecBuf)。
    * **存储单元 (Store Unit):** 负责将数据 *从* 向量缓存 (VecBuf) 传输回 *到* GlobalMem。
    * **向量计算单元 (Vector Compute Unit):** 对驻留在向量缓存 (VecBuf) 中的数据执行实际计算 (如向量加法、乘法、规约)。可能仅支持连续或带掩码 (masked) 的访问模式。
    * **标量计算单元:** 类似一个小型 CPU，可以执行一些标量运算，用于控制流或辅助计算。

### 执行模型与并行性

- **数据流路径:** 典型数据路径是：GlobalMem → (加载单元) → VecBuf → (计算单元) → VecBuf → (存储单元) → GlobalMem。
- **核心并行性:** NPU 设计的核心思想是并行。不同的 NPU 核心可以并行工作。
- **单元并行性:** 核心内部的加载、存储和计算单元也 **可以并发操作**。例如，加载单元可以在计算单元处理当前数据块的同时，取回下一个数据块。
- **数据依赖与同步:** 对于仅使用 *一组缓冲区* 处理的数据，存在严格的串行依赖：必须先完成 `Load`，该数据的 `Compute` 才能开始；必须先完成 `Compute`，`Store` 才能写回结果。
- **同步机制:** 底层通常通过类似 `set_flag / wait_flag` 机制实现同步。如果管理不当，会导致单元空闲等待 (stalls)。

### AUL流水线表达

AUL 提供了高级抽象，简化了流水线编程：

- **`U.Pipelined(iterations=N)`:** 高级指令，告知 LLM/编译器，循环迭代期望以流水线方式执行，利用硬件的并行能力。

----------------------------------------------------------
