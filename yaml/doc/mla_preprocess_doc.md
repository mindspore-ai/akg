# mla

## 描述

Multi Latent Attention，DeepSeek模型中优化技术，使用低秩压缩方法减少kvcache的显存占用。

## 输入参数

| Name | DType | Shape  | Optional | Inplace | Format | Description |
|------|-------|--------|----------|---------|--------|-------------|
| input1 | Tensor[float16/bfloat16] | (N, 7168) | No | No | ND | 融合前rmsnorm_quant1的输入tensor |
| gamma1 | Tensor[float16/bfloat16] | (7168) | No | No | ND | 融合前rmsnorm_quant1的gamma, 数据类型与input1一致 |
| beta1 | Tensor[float16/bfloat16] | (7168) | No | No | ND | 融合前无此输入，数据类型与input1一致 |
| quant_scale1 | Tensor[float16/bfloat16] | (1) | No | No | ND | 融合前rmsnorm_quant1的quant_scale, 数据类型与input1一致 |
| quant_offset1 | Tensor[int8] | (1) | No | No | ND | 融合前rmsnorm_quant1的offset |
| wdqkv | Tensor[int8] | (1, 224, 2112, 32) | No | No | NZ | 融合前QuantBatchMatmul1的权重，qkv的权重 |
| bias1 | Tensor[int32] | (2112) | No | No | ND | 融合前QuantBatchMatmul1的bias |
| de_scale1 | Tensor[float32/int64] | (2112) | No | No | ND | 融合前QuantBatchMatmul1的deScale, 输入是float16时，是int64类型；输入是bfloat16时，输入是float32 |
| gamma2 | Tensor[float16/bfloat16] | (1536) | No | No | ND | 融合前rmsnorm_quant2的gamma, 数据类型与input1一致 |
| beta2 | Tensor[float16/bfloat16] | (1536) | No | No | ND | 融合前无此输入，数据类型与input1一致 |
| quant_scale2 | Tensor[float16/bfloat16] | (1) | No | No | ND | 融合前rmsnorm_quant2的quant_scale, 数据类型与input1一致 |
| quant_offset2 | Tensor[int8] | (1) | No | No | ND | 融合前rmsnorm_quant2的offset |
| wuq | Tensor[int8] | (1, 48, headNum*192, 32) | No | No | NZ | 融合前QuantBatchMatmul2的权重，qkv的权重 |
| bias2 | Tensor[int32] | (2112) | No | No | ND | 融合前QuantBatchMatmul2的bias |
| de_scale2 | Tensor[float32/int64] | (2112) | No | No | ND | 融合前QuantBatchMatmul2的deScale, 输入是float16时，是int64类型；输入是bfloat16时，输入是float32 |
| gamma3 | Tensor[float16/bfloat16] | (512) | No | No | ND | 融合前rmsnorm的gamma, 数据类型与input1一致 |
| sin1 | Tensor[float16/bfloat16] | (tokenNum, 64) | No | No | ND | 融合前rope输入 |
| cos1 | Tensor[float16/bfloat16] | (tokenNum, 64) | No | No | ND | 融合前rope输入 |
| sin2 | Tensor[float16/bfloat16] | (tokenNum, 64) | No | No | ND | 融合前rope输入 |
| cos2 | Tensor[float16/bfloat16] | (tokenNum, 64) | No | No | ND | 融合前rope输入 |
| wuk | Tensor[float16/bfloat16] | (headNum, 128, 512) | No | No | ND | 融合前batchMatmul的权重，k的权重 |
| key_cache | Tensor[float16/bfloat16/int8] | cache_mode=0 (blockNum, blockSize, 1, 576) | No | Yes | ND | 当cache_mode=0时，kv和q拼接后输出 |
|           |                               | cache_mode=1 (blockNum, blockSize, 1, 512) | No | Yes | ND | 当cache_mode=1时，拆分成krope和ctkv |
|           |                               | cache_mode=2 (blockNum, 1*512/32， blockSize, 32) | No | Yes | NZ | 当cache_mode=2时，krope和ctkv NZ输出， ctkv和qnope量化 |
|           |                               | cache_mode=3 (blockNum, 1*512/16， blockSize, 16) | No | Yes | NZ | 当cache_mode=3时，krope和ctkv NZ输出 |
| krope_cache | Tensor[float16/bfloat16] | cache_mode=1 (blockNum, blockSize, 1, 64) | No | Yes | ND | 当cache_mode=1时，拆分成krope和ctkv |
|             |                          | cache_mode=2或3 (blockNum, 1*64/16， blockSize, 16) | No | Yes | ND | |
| slot_mapping | Tensor[int32] | (tokenNum) | No | No | ND | 融合前reshape_and_cache的blocktable |
| ctkv_scale | Tensor[float16/bfloat16] | (1) | No | No | ND | cache_mode=2时，作为量化的scale |
| qnope_scale | Tensor[float16/bfloat16] | (headNum) | No | No | ND | cache_mode=2时，作为量化的scale |
| cache_mode | int | / | / | / | / | 详见key_cache描述 |

## 输出参数

| Name   | DType           | Shape                                | Description |
|--------|-----------------|--------------------------------------|-------------|
| q_out | Tensor[float16/bfloat16/int8] | cache_mode=0时 (num_tokens, num_heads, 576) | |
|       |                               | cache_mode=1或2或3时 (num_tokens, num_heads, 512) | |
| key_cache | Tensor[float16/bfloat16/int8] | 同key_cache | inplace更新，同同key_cache |
| qrope | Tensor[float16/bfloat16] | cache_mode=0时 (num_tokens, num_heads, 64) | cache_mode=0时无此输出 |
| krope | Tensor[float16/bfloat16] | 同krope_cache | inplace更新，同krope_cache |

## 使用示例

```python
import mindspore as ms
import ms_custom_ops
import numpy as np

# nd -> nz 
def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align

def transdata(nd_mat, block_size: tuple = (16, 16)):
    """nd to nz"""
    r, c = nd_mat.shape
    r_rounded = round_up(r, block_size[0])
    c_rounded = round_up(c, block_size[1])
    r_pad = r_rounded - r
    c_pad = c_rounded - c
    nd_mat_padded = np.pad(nd_mat, (((0, r_pad), (0, c_pad))), mode='constant', constant_values=0)
    reshaped = np.reshape(nd_mat_padded, (r_rounded // block_size[0], block_size[0], c_rounded // block_size[1],
                                          block_size[1]))
    permuted = np.transpose(reshaped, (2, 0, 1, 3))
    nz_mat = np.reshape(permuted, (permuted.shape[0], permuted.shape[1] * permuted.shape[2], permuted.shape[3]))
    return nz_mat
    
# param
n = 32
hidden_strate = 7168
head_num = 32
block_num = 32
block_size = 64
headdim = 576
data_type = ms.bfloat16
cache_mode = 1

input1 = Tensor(np.random.uniform(-2.0, 2.0, size=(n, 7168))).astype(data_type)
gamma1 = Tensor(np.random.uniform(-1.0, 1.0, size=(hidden_strate))).astype(data_type)
quant_scale1 = Tensor(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
quant_offset1 = Tensor(np.random.uniform(-128.0, 127.0, size=(1))).astype(ms.int8)
wdqkv = Tensor(np.random.uniform(-2.0, 2.0, size=(2112, 7168))).astype(ms.int8)
de_scale1 = Tensor(np.random.rand(2112).astype(np.float32) / 1000)
de_scale2 = Tensor(np.random.rand(head_num * 192).astype(np.float32) / 1000)
gamma2 = Tensor(np.random.uniform(-1.0, 1.0, size=(1536))).astype(data_type)
quant_scale2 = Tensor(np.random.uniform(-2.0, 2.0, size=(1))).astype(data_type)
quant_offset2 = Tensor(np.random.uniform(-128.0, 127.0, size=(1))).astype(ms.int8)
wuq = Tensor(np.random.uniform(-2.0, 2.0, size=(head_num * 192, 1536))).astype(ms.int8)
gamma3 = Tensor(np.random.uniform(-1.0, 1.0, size=(512))).astype(data_type)
sin1 = Tensor(np.random.uniform(-1.0, 1.0, size=(n, 64))).astype(data_type)
cos1 = Tensor(np.random.uniform(-1.0, 1.0, size=(n, 64))).astype(data_type)
sin2 = Tensor(np.random.uniform(-1.0, 1.0, size=(n, 64))).astype(data_type)
cos2 = Tensor(np.random.uniform(-1.0, 1.0, size=(n, 64))).astype(data_type)
if cache_mode == 0:
    key_cache = Tensor(np.random.uniform(-1.0, 1.0, size=(block_num, block_size, 1, headdim))).astype(data_type)
elif cache_mode in (1, 3):
    key_cache = Tensor(np.random.uniform(-1.0, 1.0, size=(block_num, block_size, 1, 512))).astype(data_type)
else:
    key_cache = Tensor(np.random.uniform(-128.0, 127.0, size=(block_num, block_size, 1, 512))).astype(ms.int8)
krope_cache = Tensor(np.random.uniform(-1.0, 1.0, size=(block_num, block_size, 1, 64))).astype(data_type)
slot_mapping = Tensor(np.random.choice(block_num * block_size, n, replace=False).astype(np.int32)).astype(ms.int32)
wuk = Tensor(np.random.uniform(-2.0, 2.0, size=(head_num, 128, 512))).astype(data_type)
bias1 = Tensor(np.random.randint(-10, 10, (1, 2112)).astype(np.int32)).astype(ms.int32)
bias2 = Tensor(np.random.randint(-10, 10, (1, head_num * 192)).astype(np.int32)).astype(ms.int32)
beta1 = Tensor(np.random.randint(-2, 2, (hidden_strate)).astype(np.float16)).astype(data_type)
beta2 = Tensor(np.random.randint(-2, 2, (1536)).astype(np.float16)).astype(data_type)
quant_scale3 = Tensor(np.random.uniform(-2.0, 2.0, size=(1))).astype(data_type)
qnope_scale = Tensor(np.random.uniform(-1.0, 1.0, size=(1, head_num, 1))).astype(data_type)
key_cache_para = Parameter(key_cache, name="key_cache")
krope_cache_para = Parameter(krope_cache, name="krope_cache")

return ms_custom_ops.mla(  
    input1,
    gamma1,
    beta1,
    quant_scale1,
    quant_offset1,
    Tensor(transdata(wdqkv.asnumpy(), (16, 32))),
    bias1,
    gamma2,
    beta2,
    quant_scale2,
    quant_offset2,
    gamma3,
    sin1,
    cos1,
    sin2,
    cos2,
    key_cache_para,
    slot_mapping,
    Tensor(transdata(wuq.asnumpy(), (16, 32))),
    bias2,
    wuk,
    de_scale1,
    de_scale2,
    quant_scale3,
    qnope_scale,
    krope_cache_para,
    cache_mode)
```
