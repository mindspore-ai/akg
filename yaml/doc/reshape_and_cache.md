# reshape_and_cache算子

## 描述

reshape_and_cache算子用于将key和value张量重塑并缓存到指定的cache张量中，支持ND和NZ两种数据格式。

## 输入参数

| Name                   | DType                             | Shape                                                                                                                   | Optional | Inplace | Format | Description                    |
|------------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------|----------|---------|--------|--------------------------------|
| key                    | Tensor[float16/bfloat16/int8]     | (num_tokens, num_head, head_dim)                                                                                        | No       | No      | ND     | key 张量                       |
| value                  | Tensor[float16/bfloat16/int8]     | (num_tokens, num_head, head_dim)                                                                                        | Yes      | No      | ND     | value 张量                     |
| key_cache              | Tensor[float16/bfloat16/int8]     | ND: (num_blocks, block_size, num_head, head_dim)<br>NZ: host_shape:((num_blocks, block_size, num_head * head_dim) <br> device_shape: (num_blocks, block_size, num_head * head_dim//16, 16, 16) [fp16/bf16]<br>(num_blocks, block_size, num_head * head_dim//32, 32, 32) [int8] | No       | Yes     | ND/NZ  | key_cache 张量                 |
| value_cache            | Tensor[float16/bfloat16/int8]     | ND: (num_blocks, block_size, num_head, head_dim)<br>NZ: host_shape:((num_blocks, block_size, num_head * head_dim) <br> device_shape: (num_blocks, block_size, num_head * head_dim//16, 16, 16) [fp16/bf16]<br>(num_blocks, block_size, num_head * head_dim//32, 32, 32) [int8]| Yes      | Yes     | ND/NZ  | value_cache 张量               |
| slot_mapping           | Tensor[int32]                      | (num_tokens,)                                                                                                           | No       | No      | ND     | slot_mapping 张量              |
| cache_mode             | int                                | -                                                                                                                       | No       | -       | -      | 缓存模式：0 表示 ND 格式，1 表示 NZ 格式 |
| head_num               | int                                | -                                                                                                                       | Yes      | -       | -      | NZ 格式时必须提供              |

## 输出参数

| Name   | DType           | Shape                                | Description |
|--------|-----------------|--------------------------------------|-------------|
| - | - | -     | 仅用于占位，无实际意义    |

## 使用示例

```python
import mindspore as ms
import ms_custom_ops

# 创建输入张量
key = ms.Tensor(np.random.rand(128, 32, 64), ms.float16)
value = ms.Tensor(np.random.rand(128, 32, 64), ms.float16)
key_cache = ms.Tensor(np.random.rand(1024, 16, 32, 64), ms.float16)
value_cache = ms.Tensor(np.random.rand(1024, 16, 32, 64), ms.float16)
slot_mapping = ms.Tensor(np.arange(128), ms.int32)

# 调用算子
_ = ms_custom_ops.reshape_and_cache(
    key=key,
    value=value,
    key_cache=key_cache,
    value_cache=value_cache,
    slot_mapping=slot_mapping,
    cache_mode=0,  # ND格式
    head_num=32
)
```