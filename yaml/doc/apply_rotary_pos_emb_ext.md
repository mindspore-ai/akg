# apply_rotary_pos_emb_ext算子

## 描述

apply_rotary_pos_emb_ext算子用于计算旋转编码操作。该算子底层调用的是aclnnApplyRotaryPosEmbV2算子。

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| query               | Tensor          | 4维[batch_size, seq_len, q_head_num, head_dim] | No       | No      | ND     | 执行旋转位置编码的第一个变量                              |
| key                 | Tensor          | 4维[batch_size, seq_len, k_head_num, head_dim] | No       | No      | ND     | 执行旋转位置编码的第二个变量                              |  
| cos                 | Tensor          | 4维[batch_size, seq_len, 1, head_dim]          | No       | No      | ND     | 表示参与计算的位置编码张量                                |
| sin                 | Tensor          | 4维[batch_size, seq_len, 1, head_dim]          | No       | No      | ND     | 表示参与计算的位置编码张量                                |
| layout              | string          | No                                     | Yes      | No      | string | 表示输入Tensor的布局格式                                 |
| rotary_mode         | string          | No                                     | Yes      | No      | string | 表示支持计算公式中的旋转模式                              |

Note:
head_dim当前只支持128.
910B/910C机器上:
rotary_mode只支持"half".
layout只支持"BSND".
query shape为[batch_size, seq_len, q_head_num, head_dim]. 支持类型为:BF16/FP16/FP32.
key shape大小为[batch_size, seq_len, k_head_num, head_dim].支持类型为:BF16/FP16/FP32.
cos/sin shape大小为[batch_size, seq_len, 1, head_dim].支持类型为:BF16/FP16/FP32.

Atlas推理机器上:
rotary_mode只支持"half".
layout只支持"BSND".
query shape为[batch_size, seq_len, q_head_num, head_dim]. 支持类型为:FP16/FP32.
key shape大小为[batch_size, seq_len, k_head_num, head_dim].支持类型为:FP16/FP32.
cos/sin shape大小为[batch_size, seq_len, 1, head_dim].支持类型为:FP16/FP32.

此外注意，ub_required = (q_n + k_n) * 128 * castSize * 2 + 128 * DtypeSize * 4 + (q_n + k_n) * 128 * castSize + (q_n + k_n) * 128 * castSize * 2 + cast * (128 * 4 * 2)， 当计算出ub_required的大小超过当前AI处理器的UB空间总大小时，不支持使用该融合算子.
不支持空tensor场景.

## 输出参数

| Name   | DType      | Shape      | Description           |
|--------|------------|------------|-----------------------|
| query_emb| Tensor   | [batch_size, seq_len, q_head_num, head_dim] | query旋转位置编码后的结果 |
| key_emb | Tensor    | [batch_size, seq_len, k_head_num, head_dim] | key旋转位置编码后的结果 |

query_emb数据类型和query相同，shape大小一样。
key_emb数据类型和key相同，shape大小一样。

更多详细信息请参考：[aclnnApplyRotaryPosEmbV2](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/context/aclnnApplyRotaryPosEmbV2.md)


## 特殊说明

## 使用示例

### 基本使用示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_device("Ascend")

@ms.jit
def apply_rotary_pos_emb_ext_func(query, key, cos, sin, layout="BSND", rotary_mode="half"):
   return ms_custom_ops.apply_rotary_pos_emb_ext(query, key, cos, sin, layout, rotary_mode)

batch = 1
seq_len = 1
q_num_head = 1
k_num_head = 1
head_dim = 128
query_dtype = np.float16
query_data = np.random.uniform(
        0, 1, [batch_size, seq_len, num_head, hidden_dim]
    ).astype(query_dtype)
key_data = np.random.uniform(
        0, 1, [batch_size, seq_len, num_head, hidden_dim]
    ).astype(query_dtype)
cos_data = np.random.uniform(0, 1, [batch_size, seq_len, 1, hidden_dim]).astype(
        query_dtype
    )
sin_data = cos_data = np.random.uniform(
        0, 1, [batch_size, seq_len, 1, hidden_dim]
    ).astype(query_dtype)

query = Tensor(query_data, dtype=get_ms_dtype(query_dtype))
key = Tensor(key_data, dtype=get_ms_dtype(query_dtype))
cos = Tensor(cos_data, dtype=get_ms_dtype(query_dtype))
sin = Tensor(sin_data, dtype=get_ms_dtype(query_dtype))

query_emb, key_emb = apply_rotary_pos_emb_ext_func(query, key, cos, sin)
```
