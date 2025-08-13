# type_cast算子

## 描述

type_cast算子用于在`int8`与`qint4x2`两种数据类型之间进行相互转换。注意：当输入为`int8`时，数据已按`int4`的内存布局存放，一个`int8`中包含两个`int4`数据。

## 输入参数

| Name         | DType           | Shape  | Optional | Inplace | Format | Description                                 |
|--------------|-----------------|--------|----------|---------|--------|---------------------------------------------|
| x            | Tensor[int8/qint4x2] | 任意   | No       | No      | ND     | 需要转换的输入张量                           |
| output_dtype | dtype.Number    | -      | No       | -       | -      | 目标数据类型，仅支持`ms.int8`与`ms.qint4x2` |

## 输出参数

| Name   | DType           | Shape      | Description |
|--------|-----------------|------------|-------------|
| output | int8/qint4x2    | 与`x`相同 | 转换后的输出张量 |

## 使用示例

```python
import mindspore as ms
import ms_custom_ops
import numpy as np

# 构造示例输入（按int4布局打包到int8）
x_np = np.random.randn(3, 4).astype(np.int8)
x_int4 = x_np.reshape(-1) & 0x000F
x_int4 = x_int4[0::2] | (x_int4[1::2] << 4)
x_int4 = x_int4.reshape(3, 2)
x = ms.Tensor(x_int4, ms.int8)

# 将int8(打包int4x2)转换为qint4x2
output = ms_custom_ops.type_cast(x, ms.qint4x2)
print(output.dtype)
# Int4
```


