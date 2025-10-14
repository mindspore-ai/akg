# apply_rotary_pos_emb_v3算子

## 描述

apply_rotary_pos_emb_v3算子用于计算旋转编码操作。且支持部分数据参与选择位置编码计算。

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| query               | Tensor(dtype=FP16/FP32)          | 3维[tokens, q_head_num, qk_head_dim] | No       | Yes      | ND     | 执行旋转位置编码的第一个变量                              |
| key                 | Tensor(dtype=FP16/FP32)          | 3维[tokens, k_head_num, qk_head_dim] | No       | Yes      | ND     | 执行旋转位置编码的第二个变量                              |  
| cos                 | Tensor(dtype=FP16/FP32)          | 2维[tokens, cos_sin_head_dim]          | No       | No      | ND     | 表示参与计算的位置编码张量                                |
| sin                 | Tensor(dtype=FP16/FP32)          | 2维[tokens, cos_sin_head_dim]          | No       | No      | ND     | 表示参与计算的位置编码张量                                |
| layout              | string          | No                                     | Yes      | No      | string | 表示输入Tensor的布局格式                    |
| rotary_mode         | string          | No                                     | Yes      | No      | string | 表示支持计算公式中的旋转模式                              |

Note:
- 产品支持: Atlas推理系列产品AI Core
- rotary_mode: 当前仅支持'interleave'模式
- layout: 当前仅支持'BSH'
- dtype: query/key/cos/sin数据类型支持FP16/FP32，且四个输入参数类型一致。
- head_dim: 令`rotary_head_dim = 2 * cos_sin_head_dim`。
  - 要求`qk_head_dim >= rotary_head_dim`, qk_head_dim 不能小于rotary_head_dim。
  - 当`qk_head_dim > rotary_head_dim`时，只对`query/key[...:rotary_head_dim]` 做旋转位置编码。且（qk_head_dim - rotary_head_dim）* size(dtype)必须能被32整除
  - cos_sin_head_dim * sizeof(dtype) 必须能被32整除


## 输出参数

| Name   | DType      | Shape      | Description           |
|--------|------------|------------|-----------------------|
| query_emb| Tensor   | [tokens, q_head_num, qk_head_dim] | query旋转位置编码后的结果 |
| key_emb | Tensor    | [tokens, q_head_num, qk_head_dim] | key旋转位置编码后的结果 |

query_emb数据类型和query相同，shape大小一样。
key_emb数据类型和key相同，shape大小一样。


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

tokens = 4096
head_num_q = 32
head_num_k =2
qk_head_dim = 128
rotary_head_dim = 64
layout='BSH'
rotary_mode='interleave'
cos_head_dim = rotary_head_dim // 2
query_dtype=ms.float16
np_query = np.random.random((tokens, head_num_q, qk_head_dim))
np_key = np.random.random((tokens, head_num_k, qk_head_dim))
np_cos = np.random.random((tokens, cos_head_dim))
np_sin = np.random.random((tokens, cos_head_dim))
query = Tensor(np_query, dtype=query_dtype)
key = Tensor(np_key , dtype=query_dtype)
cos = Tensor(np_cos, dtype=query_dtype)
sin = Tensor(np_sin, dtype=query_dtype)
out_query, out_key = ms_custom_ops.apply_rotary_pos_emb_v3(query, key, cos, sin, layout, rotary_mode)
```
