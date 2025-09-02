# mla

## 描述

Multi Latent Attention，DeepSeek模型中优化技术，使用低秩压缩方法减少kvcache的显存占用。

## 输入参数

| Name             | DType                         | Shape                                                                                                                   | Optional | Inplace | Format | Description                                              |
|------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------|----------|---------|--------|----------------------------------------------------------|
| q_nope           | Tensor[float16/bfloat16/int8] | (num_tokens, num_heads, 512)                                                                                            | No       | No      | ND     | 查询向量中不参与位置编码计算的部分                       |
| q_rope           | Tensor[float16/bfloat16]      | (num_tokens, num_heads, 64)                                                                                             | No       | No      | ND     | 查询向量中参与位置编码计算的部分                         |
| ctkv             | Tensor[float16/bfloat16/int8] | ND: (num_blocks, block_size, kv_heads, 512)<br>NZ并且数据类型为int8: (num_blocks, kv_heads*512/32, block_size, 32)<br>NZ并且数据类型为bfloat16/float16：(num_blocks, kv_heads*512/16, block_size, 16) | No       | No      | ND/NZ  | key/value缓存，不包含位置编码计算，数据类型为int8时，数据排布必须是NZ |
| block_tables     | Tensor[int32]                 | ND: (batch, max_num_blocks_per_query)                                                                                   | No       | No      | ND     | 每个query的kvcache的block映射表                          |
| attn_mask        | Tensor[float16/bfloat16]      | mask_type为1：(num_tokens, max_seq_len)<br>mask_type为2：(125 + 2 * aseqlen, 128)                                        | Yes      | No      | ND     | 注意力掩码，mask_type不为0时需要传入                     |
| deq_scale_qk     | Tensor[float]                 | (num_heads)                                                                                                             | Yes      | No      | ND     | 用于qnope per_head静态对称量化，当kvcache为NZ并且数据类型为int8时需要传入 |
| deq_scale_pv     | Tensor[float]                 | (num_heads)                                                                                                             | Yes      | No      | ND     | 用于ctkv per_head静态对称量化，当kvcache为NZ并且数据类型为int8时需要传入 |
| q_seq_lens       | Tensor[int32]                 | ND: (batch)                                                                                                             | No       | No      | ND     | 每个batch对应的query长度，取值范围[1, 4]。需要CPU Tensor   |
| context_lens     | Tensor[int32]                 | ND: (batch)                                                                                                             | No       | No      | ND     | 每个batch对应的kv长度。需要CPU Tensor                    |
| head_num         | int                           | -                                                                                                                       | Yes      | -       | -      | query头数量，取值范围{8, 16, 32, 64, 128}，默认值32       |
| scale_value      | float                         | -                                                                                                                       | Yes      | -       | -      | Q*K后的缩放系数，取值范围(0, 1]                          |
| kv_head_num      | int                           | -                                                                                                                       | Yes      | -       | -      | kv头数量，当前只支持取值1，默认值1                       |
| mask_type        | int                           | -                                                                                                                       | Yes      | -       | -      | mask类型，取值：0-无mask；1-并行解码mask；2：传入固定shape的mask。默认值为0 |
| input_format     | int                           | -                                                                                                                       | Yes      | -       | -      | 指定ctkv和k_rope的输入排布格式：0-ND；1-NZ。默认值为0     |
| is_ring          | int                           | -                                                                                                                       | Yes      | -       | -      | 预留字段，当前取值为0                                    |

## 输出参数

| Name   | DType           | Shape                                | Description |
|--------|-----------------|--------------------------------------|-------------|
| attention_out | Tensor[float16/bfloat16] | (num_tokens, num_heads, 512)     | Attention计算输出    |
| lse | Tensor[float16/bfloat16] | (num_tokens, num_heads, 1)     | 预留字段，lse输出，当前输出无效值    |

## 使用示例

```python
import mindspore as ms
import ms_custom_ops
import numpy as np

batch = 4
num_tokens = 5
num_heads = 32
num_blocks = 1024
block_size = 128
kv_heads = 1

# 创建queyr和kvcache
np_q_nope = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, 512))
np_q_rope = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, 64))
np_ctkv = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, 512))
np_k_rope = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, 64))
q_nope_tensor = Tensor(np_q_nope, dtype=ms.bfloat16)
q_rope_tensor = Tensor(np_q_rope, dtype=ms.bfloat16)
ctkv_tensor = ms.Parameter(Tensor(np_ctkv, dtype=ms.bfloat16), name="ctkv")
k_rope_tensor = ms.Parameter(Tensor(np_k_rope, dtype=ms.bfloat16), name="k_rope")

# 创建sequence length
np_context_lens = np.array([192, 193, 194, 195]).astype(np.int32)
np_q_seq_lens = np.array([1, 1, 1, 2]).astype(np.int32)
q_seq_lens_tensor = Tensor(np_q_seq_lens)
context_lengths_tensor = Tensor(np_context_lens)

max_context_len = max(np_context_lens)
max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size

# 创建block table
block_tables_list = []
for i in range(num_tokens):
    block_table = [i * max_num_blocks_per_seq + _ for _ in range(max_num_blocks_per_seq)]
    block_tables_list.append(block_table)
block_tables_tensor = Tensor(np.array(block_tables_list).astype(np.int32))

# 创建并行解码mask
pre_qseqlen = 0
np_mask = np.zeros(shape=(num_tokens, max_context_len)).astype(np.float32)
for i in range(batch):
    qseqlen = np_q_seq_lens[i]
    kseqlen = np_context_lengths[i]
    tri = np.ones((qseqlen, qseqlen))
    tri = np.triu(tri, 1)
    tri *= -10000.0
    np_mask[pre_qseqlen : (pre_qseqlen + qseqlen), kseqlen-qseqlen : kseqlen] = tri
    pre_qseqlen += qseqlen
mask_tensor = Tensor(np_mask, dtype=ms.bfloat16)

q_lens_cpu = q_seq_lens_tensor.move_to("CPU")
kv_lens_cpu = context_lengths_tensor.move_to("CPU")

return ms_custom_ops.mla(q_nope_tensor, q_rope_tensor, ctkv_tensor, k_rope_tensor, block_tables_tensor,
                         mask_tensor, None, None, q_lens_cpu, kv_lens_cpu, num_heads, 0.1, kv_heads, 1)
```
