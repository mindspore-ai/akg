# 第二部分：AUL语法及执行方式

## 算子函数签名

```python
# 标准AUL算子函数签名
@sub_kernel
def operator_name_kernel(input1: U.TensorPtr, input2: U.TensorPtr, output: U.TensorPtr):
    # 函数体
```
- @sub_kernel标注目前函数是算子kernel，'core_num=core_num'标注算子启用的核数
- `input/output`: Tensor数据输入输出首地址
- AUL文件中可以包含一个或多个kernel

**注意：** 用到的Shape信息直接硬编码写入AUL函数主体

## host侧调用签名

```
def operator_name(input1: U.TensorPtr, input2: U.TensorPtr, output: U.TensorPtr):
    operator_name_kernel(input1, input2, output)
```

## 基础操作函数

```python
# 创建Tile示例
tile = U.Tile(shape=(M, N, K), dtype=U.dtype)

# 数据搬移操作示例
U.data_copy(dst=dst_buffer, src=src_buffer)

# 数据填充示例
scalar_127 = U.FilledTile((M, N), U.float32, U.VecBuf, value=127.0)

# host侧创建GlobalTensor示例
global_ptr = U.SetGlobalTensorPtr(bufferSize=bufferSize)
# 等价形式
global_ptr = U.SetGlobalTensorPtr(length, dtype)
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
|  | `U.SetGlobalTensorPtr(length, dtype)` | 创建gm数据块 |
| **计算操作** | `+`, `-`, `*`, `/` | 基础算术操作 |
|  | `==`, `!=`, `>`, `<`, `>=`, `<=` | 比较操作 |
| **索引操作** | `[]` | 数据索引 |

## 编程规范

1. **先设计算法**：确定通用AUL逻辑
2. **再硬件优化**：添加目标硬件AUL优化
3. **保持简洁**：避免冗余代码
4. **正确用内存**：明确数据移动路径
5. **活用操作函数**：用`op`参数指定操作

**特别注意：** AUL为LLM设计，是描述性语言而非执行语言。目的是简化算子描述，减少平台差异，保留优化能力。

----------------------------------------------------------
