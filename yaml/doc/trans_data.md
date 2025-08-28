# trans_data算子

## 描述

trans_data算子用于进行数据格式转换，支持ND格式与FRACTAL_NZ格式之间的相互转换，主要用于深度学习模型中的张量格式适配。

## 输入参数

| Name                   | DType           | Shape                                                                    | Description                    |
|------------------------|-----------------|--------------------------------------------------------------------------|--------------------------------|
| input                  | Tensor[float16/bfloat16/int8] | 任意形状                                                   | 输入张量                       |
| transdata_type         | int             | -                                                                        | 转换类型                       |
|                        |                 |                                                                          | 0: FRACTAL_NZ_TO_ND           |
|                        |                 |                                                                          | 1: ND_TO_FRACTAL_NZ           |

## 输出参数

| Name   | DType           | Shape                                | Description |
|--------|-----------------|--------------------------------------|-------------|
| output | Tensor[float16/bfloat16/int8] | 与输入相同或根据转换规则调整的形状   | 转换后的张量    |

## 功能说明

### 转换类型说明

1. **ND_TO_FRACTAL_NZ (1)**：将ND格式张量转换为FRACTAL_NZ格式
   - 适用于需要加速计算的场景
   - 将张量重新组织为分块的内存布局

2. **FRACTAL_NZ_TO_ND (0)**：将FRACTAL_NZ格式张量转换为ND格式
   - 适用于需要标准张量操作的场景
   - 将分块的内存布局恢复为连续的ND格式

### 重要特性说明

#### 数据对齐规则

**对齐常量**：
- float16/bfloat16: 16字节对齐
- int8: 32字节对齐 (仅限ND_TO_FRACTAL_NZ)
- H维度: 始终16字节对齐 (DEFAULT_ALIGN)

**形状转换公式**：
```
ND转FRACTAL_NZ (以3D输入为例):
原始: [batch, H, W]
辅助: [batch, RoundUp(H, 16), RoundUp(W, align)/align, align]
最终: [batch, RoundUp(W, align)/align, RoundUp(H, 16), align]

其中 align = 16 (float16/bf16) 或 32 (int8)
```

## 使用示例

### 基本用法

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

# 创建输入张量
input_data = ms.Tensor(np.random.rand(2, 16, 16), ms.float16)

# ND到FRACTAL_NZ转换
output_nz = ms_custom_ops.trans_data(
    input=input_data,
    transdata_type=1  # ND_TO_FRACTAL_NZ
)

# FRACTAL_NZ到ND转换 (自动处理形状恢复)
output_nd = ms_custom_ops.trans_data(
    input=output_nz,
    transdata_type=0  # FRACTAL_NZ_TO_ND
)
```

### 完整的往返转换示例

展示自动形状恢复功能：

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

# 原始ND张量 - 注意非对齐的维度
original_shape = [2, 23, 257]  # H=23, W=257 都不是16的倍数
input_data = ms.Tensor(np.random.rand(*original_shape), ms.float16)
print(f"原始形状: {input_data.shape}")  # [2, 23, 257]

# 步骤1: ND → FRACTAL_NZ
nz_tensor = ms_custom_ops.trans_data(input=input_data, transdata_type=1)
print(f"FRACTAL_NZ形状: {nz_tensor.shape}")  # 预期: [2, 17, 32, 16]
# 注意: 23→32 (填充), 257→272→17*16 (填充后分块)

# 步骤2: FRACTAL_NZ → ND (自动恢复原始形状)
recovered_tensor = ms_custom_ops.trans_data(
    input=nz_tensor, 
    transdata_type=0  # FRACTAL_NZ_TO_ND
)
print(f"恢复的ND形状: {recovered_tensor.shape}")  # [2, 23, 257] ✅

# 验证形状是否完全恢复
assert recovered_tensor.shape == input_data.shape, "形状恢复失败！"
print("✅ 往返转换成功！形状完全恢复")
```

### 形状推断示例

根据真实实现，不同输入维度的转换规则：

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

# 2D输入: (m, n) -> NZ: (1, n_aligned/16, m_aligned, 16)
input_2d = ms.Tensor(np.random.rand(100, 257), ms.float16)
output_2d = ms_custom_ops.trans_data(input=input_2d, transdata_type=1)
# 预期输出形状: (1, 17, 112, 16) 对于float16

# 3D输入: (b, m, n) -> NZ: (b, n_aligned/16, m_aligned, 16)  
input_3d = ms.Tensor(np.random.rand(8, 100, 257), ms.float16)
output_3d = ms_custom_ops.trans_data(input=input_3d, transdata_type=1)
# 预期输出形状: (8, 17, 112, 16) 对于float16
```

### 数据类型对齐示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

# int8数据类型 (32字节对齐)
input_int8 = ms.Tensor(np.random.randint(0, 127, (1, 23, 257), dtype=np.int8))
output_int8 = ms_custom_ops.trans_data(input=input_int8, transdata_type=1)
# 预期输出形状: (1, 9, 32, 32) 对于int8

# bfloat16数据类型 (16字节对齐)  
input_bf16 = ms.Tensor(np.random.rand(2, 15, 31), ms.bfloat16)
output_bf16 = ms_custom_ops.trans_data(input=input_bf16, transdata_type=1)
# 预期输出形状: (2, 2, 16, 16) 对于bfloat16
```

## 注意事项

1. **自动形状恢复**：
   - 算子内部自动处理形状恢复逻辑，用户无需关心具体实现细节
   - 内部会根据tensor的实际形状和格式信息自动推断正确的输出尺寸
   - 确保往返转换的正确性，自动恢复原始ND形状

2. **维度约束**：
   - ND_TO_FRACTAL_NZ：支持2D和3D输入，输出为4D
   - FRACTAL_NZ_TO_ND：输入必须为4D，输出为对应的2D或3D
   - 算子内部会自动验证维度合法性

3. **数据类型支持**：
   - **ND_TO_FRACTAL_NZ**: 支持float16、bfloat16和int8数据类型
   - **FRACTAL_NZ_TO_ND**: 仅支持float16和bfloat16，**不支持int8**

4. **对齐要求**：
   - 输入张量会根据数据类型自动进行内存对齐
   - float16/bfloat16使用16字节对齐，int8使用32字节对齐

5. **性能考虑**：格式转换操作涉及内存重排，应根据实际需求合理使用

6. **兼容性**：确保硬件平台支持相应的格式转换操作

## 错误处理

- 输入张量形状包含0维度时，算子会跳过执行并返回成功
- 参数类型不匹配时，会抛出相应的类型错误
- 不支持的转换类型组合会导致执行失败

## 支持的运行模式

- **Graph Mode**：支持静态图模式执行
- **PyNative Mode**：支持动态图模式执行

## 硬件要求

- **Ascend 910B**：推荐的硬件平台
- 其他Ascend系列芯片（具体支持情况请参考硬件兼容性文档）