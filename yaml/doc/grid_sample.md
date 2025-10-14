# grid_sample算子

## 描述

提供一个输入tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| input               | Tensor(dtype=FP32)          | 4维[n, h_in, w_in, c] | No       | No      | ND     | 输出tensor                              |
| grid                 | Tensor(dtype=FP32)          | 3维[tokens, k_head_num, qk_head_dim] | No       | Yes      | ND     | 执行旋转位置编码的第二个变量                              |  
| interpolation_mode   | String          | No          | No       | No      | string     | 插值模式，只支持“bilinear”                                |
| padding_mode       | String          | No         | No       | No      | string     | padding_mode, 只支持“border”                                |
| align_corners              | Bool          | No        | Yes      | No      | Bool |    表示设定特征图坐标与特征值的对应方式，设定为true时，特征值位于像素中心。设定为false时，特征值位于像素的角点。只支持false           |




## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| output | Tensor | [n, h_out, w_out, c] | 输出tensor，数据类型float32 |

## 特殊说明

## 使用示例

### 基本使用示例

```python

import mindspore as ms
import numpy as np
import ms_custom_ops
from mindspore import context, Tensor

ms.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

n_in = 1
c_in = 512
h_in = 24
w_in = 24
h_out = 64
w_out = 1
interpolation_mode = "bilinear"
padding_mode = "border"
align_corners = False
np_input = np.random.random((n_in, h_in, w_in, c_in)).astype(input_dtype)
np_grid = np.random.uniform(-1, 1, (n_in, h_out, w_out, 2)).astype(grid_dtype)
input_data = Tensor(np_input)
grid = Tensor(np_grid)
output_data = ms_custon_ops.grid_sample(input_data, grid, interpolation_mode, padding_mode, align_corners)
```
