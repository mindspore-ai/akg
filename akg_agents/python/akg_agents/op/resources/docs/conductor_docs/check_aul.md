# 第一部分：AUL规范基础定义

## 简介

AUL (AI Unity Language) 是专门为LLM辅助AI算子生成而设计的一套统一表达语言。其目的是让各类AI模型更简单、明确地设计算子算法、优化调度方案、对接多硬件后端。

## 基础数据类型

| 类型 | 描述 | 用法示例 |
|-----|-----|---------|
| TensorPtr | Tensor指针 | `input: U.TensorPtr` |
| BufferPtr | 内存指针 | `buffer: U.BufferPtr` |
| Tile | 数据块单元 | `tile = U.Tile(shape=(M, N, K), dtype=U.dtype)` |

----------------------------------------------------------


# 第二部分：AUL语法及执行方式

## 算子函数签名

```python
# 标准AUL算子函数签名
def operator_name(input1: U.TensorPtr, input2: U.TensorPtr, output: U.TensorPtr):
    # 函数体
```
- `input/output`: Tensor数据输入输出首地址

**注意：** 用到的Shape信息直接硬编码写入AUL函数主体

## 基础操作函数

```python
# 创建Tile示例
tile = U.Tile(shape=(M, N, K), dtype=U.float16)

# 数据搬移操作示例
U.data_copy(dst=dst_buffer, src=src_buffer)

# 数据填充示例
scalar_127 = U.FilledTile((M, N), U.float32, U.VecBuf, value=127.0)
```

## 基础计算操作

- 算术操作: `+`, `-`, `*`, `/`
- 比较操作: `==`, `!=`, `>`, `<`, `>=`, `<=`
- 索引操作: `tensor[start:end]`, `tile[start:end]`

**注意：** 硬件后端可扩展计算操作（见NPU上的AUL拓展等扩展部分）

## 通用AUL语法速查表

| 类别 | 语法/API | 描述 |
|------|----------|------|
| **类型** | `U.TensorPtr`, `U.BufferPtr`, `U.Tile` | 基础数据类型 |
| **数据操作** | `U.Tile(shape, dtype)` | 创建数据块 |
|  | `U.data_copy(dst, src)` | 数据搬移 |
|  | `U.FilledTile(shape, dtype, value)` | 数据填充 |
| **计算操作** | `+`, `-`, `*`, `/` | 基础算术操作 |
|  | `==`, `!=`, `>`, `<`, `>=`, `<=` | 比较操作 |
| **索引操作** | `[]` | 数据索引 |

## 编程规范

1. **先设计算法**：确定通用AUL逻辑
2. **再硬件优化**：添加目标硬件AUL优化
3. **保持简洁**：避免冗余代码
4. **正确用内存**：明确数据移动路径
5. **活用操作函数**：用`op`参数指定操作

----------------------------------------------------------


# 第三部分：NPU上的AUL拓展

## NPU内存层次

| 层次 | 描述 |
|-----|-----|
| GlobalMem | 全局内存，容量大但访问慢 |
| VecBuf | 向量缓存，用于向量计算的高速缓存 |
| CubeBuf | 矩阵乘内存，用于优化矩阵运算 |


## NPU并行控制

```python
# 获取当前核的唯一标识，范围为[0, core_num-1]
core_idx = U.get_core_idx()

# 软件流水线循环 (Software Pipelining)
# U.Pipelined 用于指示循环的迭代可以重叠执行，以隐藏延迟
# iterations 参数指定逻辑迭代的总次数
for iter_idx in U.Pipelined(iterations=LOOP_COUNT):
    # 其数据加载和计算预期会由后端进行重叠调度，不需要显式进行
    pass
```

**核心API说明：**
- `U.get_core_idx()`: 获取当前核心ID，用于多核并行处理。
- `U.Pipelined(iterations=N)`: 定义N次迭代循环。


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

----------------------------------------------------------