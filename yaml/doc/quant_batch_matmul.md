# quant_batch_matmul算子

## 描述

quant_batch_matmul算子用于执行批量量化矩阵乘法操作。该算子支持输入张量的转置、缩放、偏移和偏置等操作，并可指定输出数据类型。

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| x1                  | Tensor          | 2~6维  | No       | No      | ND     | 第一个输入矩阵的批量张量                               |
| x2                  | Tensor          | 2~6维  | No       | No      | ND/FRACTAL_NZ | 第二个输入矩阵的批量张量，支持ND格式或FRACTAL_NZ格式    |
| scale               | Tensor          | 标量或适当广播的形状                    | No       | No      | ND     | 缩放因子，用于量化/反量化过程                          |
| offset              | Tensor          | 标量或适当广播的形状                    | Yes      | No      | ND     | 偏移量，默认为None                                      |
| bias                | Tensor          | 适当广播的形状                          | Yes      | No      | ND     | 偏置张量，默认为None                                    |
| pertoken_scale | Tensor        | 适当广播的形状                          | Yes      | No      | ND     | 逐token缩放因子，默认为None                            |
| transpose_x1        | bool            | -                                      | Yes      | -       | -      | 是否对x1进行转置操作，默认为False                      |
| transpose_x2        | bool            | -                                      | Yes      | -       | -      | 是否对x2进行转置操作，默认为False                      |
| x2_format           | str             | -                                      | No       | -       | -      | x2的format格式，支持"ND"和"FRACTAL_NZ", 默认为"ND" |
| output_dtype        | dtype.Number    | -                                      | Yes      | -       | -      | 输出数据类型，支持float16、bfloat16、int8，默认为float16 |

## 输出参数

| Name   | DType      | Shape      | Description           |
|--------|------------|------------|-----------------------|
| output | Tensor     | 符合批量矩阵乘法规则的形状 | 批量矩阵乘法的计算结果 |

更多详细信息请参考：[aclnnQuantMatmulV4](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/context/aclnnQuantMatmulV4.md)


## 特殊说明

- 在PYNATIVE_MODE模式下，x2不支持FRACTAL_NZ格式。


## 使用示例

### 基本使用示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_device("Ascend")

@ms.jit
def quant_batch_matmul_func(x1, x2, scale, offset=None, bias=None, 
                            pertoken_scale=None, transpose_x1=False, 
                            transpose_x2=False, x2_format="ND", dtype=ms.float16):
   return ms_custom_ops.quant_batch_matmul(x1, x2, scale, offset, bias, 
                                           pertoken_scale, transpose_x1, 
                                           transpose_x2, x2_fromat, dtype)

batch = 2
m = 128
k = 256
n = 128
x1 = np.random.randint(-5, 5, size=(batch, m, k)).astype(np.int8)
x2 = np.random.randint(-5, 5, size=(batch, k, n)).astype(np.int8)
scale = np.ones([n]).astype(np.float32)

ms_x1 = Tensor(x1)
ms_x2 = Tensor(x2)
ms_x2 = ms_custom_ops.trans_data(ms_x2, transdata_type=1)
ms_scale = Tensor(scale)
output = quant_batch_matmul_func(ms_x1, ms_x2, ms_scale, x2_format="FRACTAL_NZ", dtype=ms.bfloat16)
```
