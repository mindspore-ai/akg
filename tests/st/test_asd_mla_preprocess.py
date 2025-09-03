# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
test_asd_mla_preprocess
"""

import os
import numpy as np
import pytest
from mindspore import Tensor, context, Parameter, jit
import mindspore as ms
from scipy.special import logsumexp
import ms_custom_ops

QUANTMAX = 127
QUANTMIN = -128

class AsdMlaPreprocessCustom(ms.nn.Cell):
    def __init__(self):
        super().__init__()
    
    @jit
    def construct(self, input1, gamma1, beta1, quant_scale1, quant_offset1, wdqkv, bias1, gamma2, beta2,
                  quant_scale2, quant_offset2, gamma3, sin1, cos1, sin2, cos2, key_cache, slot_mapping,
                  wuq, bias2, wuk, de_scale1, de_scale2, quant_scale3, qnope_scale, krope_cache_para, cache_mode):
        return ms_custom_ops.mla_preprocess(
            input1, gamma1, beta1, quant_scale1, quant_offset1, wdqkv, bias1, gamma2, beta2, quant_scale2,
            quant_offset2, gamma3, sin1, cos1, sin2, cos2, key_cache, slot_mapping, wuq, bias2, wuk, de_scale1,
            de_scale2, quant_scale3, qnope_scale, krope_cache_para, cache_mode)


def rms_norm_quant_calc(input_x, gamma, beta, quant_scale, quant_offset, epsilon):
    """
    rms norm quant calculation
    """
    out_shape = input_x.shape
    scale = 1.0 / quant_scale.item()
    input_scale = np.array(scale, dtype=np.float32)
    offset = quant_offset.item()
    input_offset = np.array(offset, dtype=np.float32)
    input0 = np.array(input_x, dtype=np.float32)
    input1 = np.array(gamma, dtype=np.float32)

    square_sum = np.sum(np.square(input0), axis=-1, keepdims=True)
    np_sqrt = np.sqrt(square_sum / out_shape[-1] + epsilon)
    factor = np.zeros_like(np_sqrt)
    for i in range(np_sqrt.shape[0]):
        factor[i] = 1.0 / np_sqrt[i]
    output = input0 * factor * input1
    output = (output + beta) * input_scale + input_offset
    output = np.round(output)
    output = output.astype(np.float16)
    output = np.minimum(output, 127)
    output = np.maximum(output, -128)
    output = output.astype(np.int8)
    return output

def rms_norm_golden(x, gamma, rms_hidden_size, epsilon):
    """
    rms norm calculation
    """
    x_float32 = x.astype(np.float32)
    square_sum = np.sum(np.square(x_float32), axis=-1, keepdims=True)
    rms = 1.0 / np.sqrt(square_sum / rms_hidden_size + epsilon)
    gamma_float32 = gamma.astype(np.float32)
    rms_norm = rms * x_float32 * gamma_float32
    result = rms_norm.astype(np.float32)
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:.15f}".format})
    return result

def rotate_half(k_temp):
    """
    rotate half calculation
    """
    first_half, second_half = np.split(k_temp, 2, axis=1)
    first_half = Tensor(first_half).astype(ms.bfloat16).astype(ms.float32).asnumpy()
    second_half = Tensor(second_half).astype(ms.bfloat16).astype(ms.float32).asnumpy()
    processed_k_split = np.concatenate((-second_half, first_half), axis=1)
    return processed_k_split

def rac_golden(key_rac, block_size, slot_mapping, key_cacheout_golden):
    """
    reshape and cache calculation
    """
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        block_index = slot // block_size
        block_offset = slot % block_size
        token_key = key_rac[i]
        key_cacheout_golden[block_index, block_offset, 0, :] = token_key[0]
    return key_cacheout_golden

def rotate_half_x(q_temp, head_num):
    """
    rotate_half_x calculation
    """
    # 将 q_temp 切分为 head_num 份
    q_splits = np.array_split(q_temp, head_num, axis=1)
    processed_q_splits = []
    for q_split in q_splits:
        # 将每个分块分成两半
        first_half, second_half = np.split(q_split, 2, axis=1)
        # 负数的操作
        processed_q_split = np.concatenate((-second_half, first_half), axis=1)
        processed_q_splits.append(processed_q_split)
    # 将所有分块拼接起来
    return np.concatenate(processed_q_splits, axis=1)

def rope_concat_golden(q, sin, cos, concat_input, input_token_num, head_num, rope_hidden_size, dtype):
    """
    rope concat calculation
    """
    pad_sin = np.tile(sin, (1, head_num))
    pad_cos = np.tile(cos, (1, head_num))
    if dtype == ms.bfloat16:
        rope_res = (Tensor(q).astype(ms.bfloat16) * Tensor(pad_cos).astype(ms.bfloat16) +
                    Tensor(rotate_half_x(q, head_num)).astype(ms.bfloat16) * Tensor(pad_sin).astype(ms.bfloat16))
        rope_res = rope_res.reshape(input_token_num, head_num, rope_hidden_size)
        rope_res = rope_res.astype(np.float32)
        result = np.concatenate((concat_input.astype(np.float32), rope_res), axis=2)
    else:
        rope_res = q * pad_cos + rotate_half_x(q, head_num) * pad_sin
        rope_res = rope_res.reshape(input_token_num, head_num, rope_hidden_size)
        rope_res = rope_res.astype(np.float16)
        result = np.concatenate((concat_input.astype(np.float16), rope_res), axis=2)
    return result

def ein_sum_out_quant_golden(input1, scale):
    """
    rope concat calculation
    """
    quant = input1.astype(np.float32) * Tensor(scale).astype(ms.bfloat16).astype(ms.float32).asnumpy()
    output = np.sign(quant) * np.floor(np.abs(quant) + 0.5).astype(np.float16)
    output = np.minimum(output, np.float16(QUANTMAX))
    output = np.maximum(output, np.float16(QUANTMIN))
    return output.astype(np.int8)

def s8_saturation(inputdata):
    inputdata = np.clip(inputdata, QUANTMIN, QUANTMAX)
    return np.rint(inputdata).astype(np.int8)

def quant_func(x, qscale):
    # qscale = qscale.to(torch.float)
    qscale = 1.0 / qscale
    x = x.astype(np.float32)
    # 使用广播机制来避免显式的循环
    scaled_values = (x * qscale).astype(np.float16)
    # 饱和+四舍五入+转int8
    s8_res_cal = s8_saturation(scaled_values)
    return s8_res_cal

def reshape_and_cache_nz(input1, key_cache, slot_mapping, num, fenxin, loop):
    """
    reshape and cache nz calculation
    """
    key_cache = key_cache.flatten()
    input_array = input1.reshape(-1, num)
    for i in range(len(slot_mapping)):
        slot_idx = slot_mapping[i]
        outer_idx = int(slot_idx / 128)
        inner_idx = slot_idx % 128
        stride = 128 * fenxin
        for j in range(loop):
            start_idx = int(inner_idx * fenxin + j * stride + outer_idx * 128 * num)
            end_idx = start_idx + fenxin
            src_start = j * fenxin
            src_end = (j + 1) * fenxin
            key_cache[start_idx:end_idx] = input_array[i][src_start:src_end]

    return key_cache

def golden_calculate(input1, gamma1, beta1, quant_scale1, quant_offset1, wdqkv, bias1, gamma2, beta2, quant_scale2,
                     quant_offset2, gamma3, sin1, cos1, sin2, cos2, key_cache, slot_mapping, wuq, bias2, wuk, de_scale1,
                     de_scale2, quant_scale3, qnope_scale, krope_cache, cache_mode, data_type):
    """
    golden calculate
    """
    epsilon = 1e-6
    n = input1.shape[0]
    head_num = wuk.shape[0]
    block_size = key_cache.shape[1]
    rms_hidden_size = 512
    rope_hidden_size = 64

    # 1. rms_norm_quant_calc
    rms_norm_quant_out1 = rms_norm_quant_calc(
        input1, gamma1, beta1, quant_scale1, quant_offset1, epsilon
    )

    # 2. matmul rmsquantout, wdqkv.transpose(0,1)
    wdqkv_transposed = np.transpose(wdqkv, (1, 0))
    qbmm_out0 = np.matmul(rms_norm_quant_out1.astype(np.float32), wdqkv_transposed.astype(np.float32))
    qbmm_out0 = qbmm_out0.astype(np.int32) + bias1
    if data_type == ms.bfloat16:
        qbmm_out0 = Tensor(qbmm_out0.astype(np.float32) * de_scale1).astype(ms.bfloat16).astype(ms.float32).asnumpy()
    else:
        qbmm_out0 = (qbmm_out0.astype(np.float32) * de_scale1).astype(np.float16)

    # SplitWithSize
    # qbmm_out0_split1, qbmm_out0_split2 = self.split_with_size(qbmm_out0, (576, 1536), -1)
    qbmm_out0_split1, qbmm_out0_split2 = np.split(qbmm_out0, [576], axis=1)

    # 3. rms_norm_quant_2 gamma2, beta2, quant_scale2, quant_offset2
    rms_norm_quant_out2 = rms_norm_quant_calc(
        qbmm_out0_split2, gamma2, beta2, quant_scale2, quant_offset2, epsilon
    )

    # 4. matmul_1 wuq de_scale2 bias2
    wuq_transposed = np.transpose(wuq, (1, 0))
    qbmm_out1 = np.matmul(rms_norm_quant_out2.astype(np.float32), wuq_transposed.astype(np.float32))
    qbmm_out1 = qbmm_out1.astype(np.int32) + bias2
    if data_type == ms.bfloat16:
        qbmm_out1 = Tensor(qbmm_out1.astype(np.float32) * de_scale2).astype(ms.bfloat16).astype(ms.float32).asnumpy()
    else:
        qbmm_out1 = (qbmm_out1.astype(np.float32) * de_scale2).astype(np.float16)

    # SplitWithSize(%37, (I64(128), I64(64)), I64(-1))
    # qbmm_out1_split1, qbmm_out1_split2 = self.split_with_size(reshape_out3, (512, 64), -1)
    qbmm_out1_split1, qbmm_out1_split2 = np.split(qbmm_out0_split1, [512], axis=1)

    # 5. rms_norm gamma3
    qbmm_out1_split1 = qbmm_out1_split1.reshape(n, 1, 512)
    rms_norm_out = rms_norm_golden(qbmm_out1_split1, gamma3, rms_hidden_size, epsilon)
    if data_type == ms.bfloat16:
        rms_norm_out = Tensor(rms_norm_out).astype(ms.bfloat16).astype(ms.float32).asnumpy()
    else:
        rms_norm_out = rms_norm_out.astype(np.float16)

    # rope cos, sin
    if data_type == ms.bfloat16:
        rope_out = (Tensor(qbmm_out1_split2).astype(ms.bfloat16) * Tensor(cos1).astype(ms.bfloat16) +
                    Tensor(rotate_half(qbmm_out1_split2)).astype(ms.bfloat16) * Tensor(sin1).astype(ms.bfloat16))
        rope_out = rope_out.astype(ms.float32).asnumpy()
        rope_out = rope_out.reshape(n, 1, rope_hidden_size)
        rope_out = Tensor(rope_out).astype(ms.bfloat16).astype(ms.float32).asnumpy()
    else:
        rope_out = qbmm_out1_split2 * cos1 + rotate_half(qbmm_out1_split2) * sin1
        rope_out = rope_out.reshape(n, 1, rope_hidden_size)
        rope_out = rope_out.astype(np.float16)

    key_rac = np.concatenate((rms_norm_out, rope_out), axis=-1)

    if cache_mode == 0:
        key_cache_copy = key_cache.copy()
        key_cache_out = rac_golden(key_rac, block_size, slot_mapping, key_cache_copy)
    elif cache_mode == 1:
        key_cache_copy = np.concatenate((key_cache, krope_cache), axis=-1)
        key_cache_out = rac_golden(key_rac, block_size, slot_mapping, key_cache_copy)

    qbmm_out1 = qbmm_out1.reshape(n, head_num, 192)
    _, mm2_out_split2 = np.split(qbmm_out1, [128], axis=2)

    # 6. bmm
    qbmm_out1_reshaped = np.transpose(qbmm_out1[:, :, :128], (1, 0, 2)).astype(np.float32)
    matmul_result = np.matmul(qbmm_out1_reshaped, wuk.astype(np.float32))
    bmm_out = np.transpose(matmul_result, (1, 0, 2))

    q_out = rope_concat_golden(mm2_out_split2.reshape(n, head_num * 64), sin2, cos2, bmm_out, n, head_num,
                               rope_hidden_size, data_type)
    if cache_mode == 0:
        return q_out, key_cache_out, None, None
    if cache_mode == 1:
        q_out0 = q_out[:, :, :512]
        q_out1 = q_out[:, :, 512:576]
        key_cache0 = key_cache_out[:, :, :, :512]
        key_cache1 = key_cache_out[:, :, :, 512:576]
        return q_out0, key_cache0, q_out1, key_cache1
    if cache_mode == 2:
        q_out0 = q_out[:, :, :512]
        q_out1 = q_out[:, :, 512:576]
        quant_test = quant_func(rms_norm_out, quant_scale3)
        key_cache0_quant = reshape_and_cache_nz(quant_test, key_cache, slot_mapping, 512, 32, 16)
        key_cache1_out = reshape_and_cache_nz(rope_out, krope_cache, slot_mapping, 64, 16, 4)
        return ein_sum_out_quant_golden(q_out0, qnope_scale), key_cache0_quant, q_out1, key_cache1_out
    if cache_mode == 3:
        q_out0 = q_out[:, :, :512]
        q_out1 = q_out[:, :, 512:576]
        key_cache0_out = reshape_and_cache_nz(rms_norm_out, key_cache, slot_mapping, 512, 16, 32)
        key_cache1_out = reshape_and_cache_nz(rope_out, krope_cache, slot_mapping, 64, 16, 4)
        return q_out0, key_cache0_out, q_out1, key_cache1_out
    print("ERROR, unsupported cache_mode!\n")
    return None, None, None, None

def kl_divergence(logits1_np, logits2_np):
    """计算 KL(p || q)，其中 p 和 q 是 log-probabilities"""
    def log_softmax(x, axis=-1):

        return x - logsumexp(x, axis=axis, keepdims=True)
    log_p = log_softmax(logits1_np, axis=-1)
    log_q = log_softmax(logits2_np, axis=-1)
    # 打印中间值进行调试
    p = np.exp(log_p)
    kl = np.where(p != 0, p * (log_p - log_q), 0.0)
    return np.sum(kl)

def cosine_similarity_numpy(vecs1, vecs2, axis=-1):
    """计算两个矩阵之间的余弦相似度"""
    norm1 = np.linalg.norm(vecs1, axis=axis, keepdims=True)
    norm2 = np.linalg.norm(vecs2, axis=axis, keepdims=True)
    dot_product = np.sum(vecs1 * vecs2, axis=axis, keepdims=True)
    cosine_sim = dot_product / (norm1 * norm2)
    return np.squeeze(cosine_sim)

def topk(v1, v2, k=5):
    """输出两个数组的 Top-K 元素索引"""
    flat_indices_v1 = np.argsort(v1, axis=None)[-k:][::-1]
    flat_indices_v2 = np.argsort(v2, axis=None)[-k:][::-1]
    print(f"GPU top-{k}: {flat_indices_v1}")
    print(f"NPU top-{k}: {flat_indices_v2}")

def compare(gpu, npu):
    """比对两个矩阵的余弦相似度"""
    gpu = gpu.flatten()
    npu = npu.flatten()
    cos = cosine_similarity_numpy(gpu, npu)
    print("Cosine Similarity:", cos)
    # 比较 Top-K
    topk(gpu, npu)
    # 判断是否通过
    if cos > 0.999:
        print("\nResult: PASS")
        return True
    print("\nResult: FAILED")
    return False

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

def mla_preprocess(n, head_num, hidden_strate, block_num, block_size, headdim, data_type, cache_mode, context_mode,
                   is_dyn=False):
    """mla preprocess main testcase function"""
    os.environ['USE_LLM_CUSTOM_MATMUL'] = "off"
    os.environ['INTERNAL_PRINT_TILING'] = "on"
    os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = ""
    os.environ["MS_ENABLE_INTERNAL_BOOST"] = "off"

    context.set_context(mode=context_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # context.set_context(save_graphs=1, save_graphs_path="./mla_preprocess_graph")

    # param
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

    if data_type == ms.bfloat16:
        q_out0_golden, key_cache0_golden, q_out1_golden, key_cache1_golden = golden_calculate(
            input1.astype(ms.float32).asnumpy(),
            gamma1.astype(ms.float32).asnumpy(),
            beta1.astype(ms.float32).asnumpy(),
            quant_scale1.astype(ms.float32).asnumpy(),
            quant_offset1.asnumpy(),
            wdqkv.asnumpy(),
            bias1.asnumpy(),
            gamma2.astype(ms.float32).asnumpy(),
            beta2.astype(ms.float32).asnumpy(),
            quant_scale2.astype(ms.float32).asnumpy(),
            quant_offset2.asnumpy(),
            gamma3.astype(ms.float32).asnumpy(),
            sin1.astype(ms.float32).asnumpy(),
            cos1.astype(ms.float32).asnumpy(),
            sin2.astype(ms.float32).asnumpy(),
            cos2.astype(ms.float32).asnumpy(),
            key_cache.astype(ms.float32).asnumpy(),
            slot_mapping.asnumpy(),
            wuq.asnumpy(),
            bias2.asnumpy(),
            wuk.astype(ms.float32).asnumpy(),
            de_scale1.asnumpy(),
            de_scale2.asnumpy(),
            quant_scale3.astype(ms.float32).asnumpy(),
            qnope_scale.astype(ms.float32).asnumpy(),
            krope_cache.astype(ms.float32).asnumpy(),
            cache_mode,
            data_type)
    else:
        q_out0_golden, key_cache0_golden, q_out1_golden, key_cache1_golden = golden_calculate(
            input1.asnumpy(),
            gamma1.asnumpy(),
            beta1.asnumpy(),
            quant_scale1.asnumpy(),
            quant_offset1.asnumpy(),
            wdqkv.asnumpy(),
            bias1.asnumpy(),
            gamma2.asnumpy(),
            beta2.asnumpy(),
            quant_scale2.asnumpy(),
            quant_offset2.asnumpy(),
            gamma3.asnumpy(),
            sin1.asnumpy(),
            cos1.asnumpy(),
            sin2.asnumpy(),
            cos2.asnumpy(),
            key_cache.asnumpy(),
            slot_mapping.asnumpy(),
            wuq.asnumpy(),
            bias2.asnumpy(),
            wuk.asnumpy(),
            de_scale1.asnumpy(),
            de_scale2.asnumpy(),
            quant_scale3.asnumpy(),
            qnope_scale.asnumpy(),
            krope_cache.asnumpy(),
            cache_mode,
            data_type)

    # expect
    net = AsdMlaPreprocessCustom()
    if data_type == ms.bfloat16:
        de_scale1 = de_scale1.astype(ms.float32)
        de_scale2 = de_scale2.astype(ms.float32)
    else:
        de_scale1 = Tensor(de_scale1.asnumpy().view(np.int32).astype(np.int64))
        de_scale2 = Tensor(de_scale2.asnumpy().view(np.int32).astype(np.int64))
    key_cache_para = Parameter(key_cache, name="key_cache")
    krope_cache_para = Parameter(krope_cache, name="krope_cache")
    if not is_dyn:
        q_out0, _, q_out1, _ = net(
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
    else:
        input1_dyn = ms.Tensor(shape=[None, None], dtype=data_type)
        gamma1_dyn = ms.Tensor(shape=[None], dtype=data_type)
        quant_scale1_dyn = ms.Tensor(shape=[None], dtype=data_type)
        quant_offset1_dyn = ms.Tensor(shape=[None], dtype=ms.int8)
        wdqkv_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.int8)

        if data_type == ms.bfloat16:
            de_scale1_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
            de_scale2_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
        else:
            de_scale1_dyn = ms.Tensor(shape=[None], dtype=ms.int64)
            de_scale2_dyn = ms.Tensor(shape=[None], dtype=ms.int64)
        gamma2_dyn = ms.Tensor(shape=[None], dtype=data_type)
        quant_scale2_dyn = ms.Tensor(shape=[None], dtype=data_type)
        quant_offset2_dyn = ms.Tensor(shape=[None], dtype=ms.int8)

        wuq_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.int8)

        gamma3_dyn = ms.Tensor(shape=[None], dtype=data_type)
        sin1_dyn = ms.Tensor(shape=[None, None], dtype=data_type)
        cos1_dyn = ms.Tensor(shape=[None, None], dtype=data_type)
        sin2_dyn = ms.Tensor(shape=[None, None], dtype=data_type)
        cos2_dyn = ms.Tensor(shape=[None, None], dtype=data_type)

        if cache_mode == 2:
            key_cache_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.int8)
        else:
            key_cache_dyn = ms.Tensor(shape=[None, None, None, None], dtype=data_type)
        krope_cache_dyn = ms.Tensor(shape=[None, None, None, None], dtype=data_type)
        slot_mapping_dyn = ms.Tensor(shape=[None], dtype=ms.int32)

        wuk_dyn = ms.Tensor(shape=[None, None, None], dtype=data_type)

        bias1_dyn = ms.Tensor(shape=[None, None], dtype=ms.int32)
        bias2_dyn = ms.Tensor(shape=[None, None], dtype=ms.int32)

        beta1_dyn = ms.Tensor(shape=[None], dtype=data_type)
        beta2_dyn = ms.Tensor(shape=[None], dtype=data_type)

        quant_scale3_dyn = ms.Tensor(shape=[None], dtype=data_type)
        qnope_scale_dyn = ms.Tensor(shape=[None, None, None], dtype=data_type)
        net.set_inputs(input1_dyn, gamma1_dyn, beta1_dyn, quant_scale1_dyn, quant_offset1_dyn, wdqkv_dyn,
                       bias1_dyn, gamma2_dyn, beta2_dyn, quant_scale2_dyn, quant_offset2_dyn, gamma3_dyn,
                       sin1_dyn, cos1_dyn, sin2_dyn, cos2_dyn, key_cache_dyn, slot_mapping_dyn, wuq_dyn,
                       bias2_dyn, wuk_dyn, de_scale1_dyn, de_scale2_dyn, quant_scale3_dyn, qnope_scale_dyn,
                       krope_cache_dyn, cache_mode)
        key_cache_para = key_cache
        krope_cache_para = krope_cache
        q_out0, _, q_out1, _ = net(
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

    if "MS_INTERNAL_ENABLE_NZ_OPS" in os.environ:
        del os.environ["MS_INTERNAL_ENABLE_NZ_OPS"]
    os.unsetenv("MS_INTERNAL_ENABLE_NZ_OPS")

    q_compare_result = False
    key_cache_compare_result = False
    if cache_mode == 0:
        q_compare_result = compare(q_out0.astype(ms.float32).asnumpy(), q_out0_golden.astype(np.float32))
        key_cache_compare_result = compare(key_cache_para.astype(ms.float32).asnumpy(),
                                           key_cache0_golden.astype(np.float32))

        assert q_compare_result and key_cache_compare_result, "q and key_cache compare failed."

    elif cache_mode in (1, 3):
        q_compare_result1 = compare(q_out0.astype(ms.float32).asnumpy(), q_out0_golden.astype(np.float32))
        q_compare_result2 = compare(q_out1.astype(ms.float32).asnumpy(), q_out1_golden.astype(np.float32))
        q_compare_result = q_compare_result1 and q_compare_result2
        key_cache_compare_result1 = compare(key_cache_para.astype(ms.float32).asnumpy(),
                                            key_cache0_golden.astype(np.float32))
        key_cache_compare_result2 = compare(krope_cache_para.astype(ms.float32).asnumpy(),
                                            key_cache1_golden.astype(np.float32))
        key_cache_compare_result = key_cache_compare_result1 and key_cache_compare_result2

        assert q_compare_result and key_cache_compare_result, "q and key_cache compare failed."

    elif cache_mode == 2:
        q_out0_diff = q_out0.asnumpy().flatten() - q_out0_golden.flatten()
        q_out0_max_diff = np.max(np.abs(q_out0_diff))
        q_compare_result1 = q_out0_max_diff <= 1
        q_compare_result2 = compare(q_out1.astype(ms.float32).asnumpy(), q_out1_golden.astype(np.float32))
        q_compare_result = q_compare_result1 and q_compare_result2

        key_cache0_diff = key_cache_para.asnumpy().flatten() - key_cache0_golden.flatten()
        key_cache0_max_diff = np.max(np.abs(key_cache0_diff))
        key_cache_compare_result1 = key_cache0_max_diff <= 1
        key_cache_compare_result2 = compare(krope_cache_para.astype(ms.float32).asnumpy(),
                                            key_cache1_golden.astype(np.float32))
        key_cache_compare_result = key_cache_compare_result1 and key_cache_compare_result2
        assert q_compare_result and key_cache_compare_result, "q and key_cache compare failed."

    else:
        print("wrong cache_mode!!!\n")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('token_num', [32])
@pytest.mark.parametrize('block_size', [64])
@pytest.mark.parametrize('block_num', [32])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [ms.bfloat16, ms.float16])
def test_mla_preprocess_cache_mode0(token_num, block_size, block_num, data_type, context_mode):
    """
    Feature: test asd_mla_preprocess operator in graph mode
    Description: test asd_mla_preprocess.
    Expectation: the result is correct
    """
    n = token_num
    head_num = 32
    hidden_strate = 7168
    block_num = block_num
    block_size = block_size
    headdim = 576
    data_type = data_type
    cache_mode = 0
    mla_preprocess(n, head_num, hidden_strate, block_num, block_size, headdim, data_type, cache_mode, context_mode,
                   is_dyn=False)

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('token_num', [32])
@pytest.mark.parametrize('block_size', [64])
@pytest.mark.parametrize('block_num', [32])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [ms.bfloat16, ms.float16])
def test_mla_preprocess_cache_mode1(token_num, block_size, block_num, data_type, context_mode):
    """
    Feature: test asd_mla_preprocess operator in graph mode
    Description: test asd_mla_preprocess.
    Expectation: the result is correct
    """
    n = token_num
    head_num = 32
    hidden_strate = 7168
    block_num = block_num
    block_size = block_size
    headdim = 576
    data_type = data_type
    cache_mode = 1
    mla_preprocess(n, head_num, hidden_strate, block_num, block_size, headdim, data_type, cache_mode, context_mode,
                   is_dyn=False)

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('token_num', [32])
@pytest.mark.parametrize('block_num', [32])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [ms.bfloat16, ms.float16])
def test_mla_preprocess_bf16_cache_mode2(token_num, block_num, data_type, context_mode):
    """
    Feature: test asd_mla_preprocess operator in graph mode
    Description: test asd_mla_preprocess.
    Expectation: the result is correct
    """
    n = token_num
    head_num = 32
    hidden_strate = 7168
    block_num = block_num
    block_size = 128
    headdim = 576
    data_type = data_type
    cache_mode = 2
    mla_preprocess(n, head_num, hidden_strate, block_num, block_size, headdim, data_type, cache_mode, context_mode,
                   is_dyn=False)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('token_num', [32])
@pytest.mark.parametrize('block_num', [32])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [ms.bfloat16, ms.float16])
def test_mla_preprocess_bf16_cache_mode3(token_num, block_num, data_type, context_mode):
    """
    Feature: test asd_mla_preprocess operator in graph mode
    Description: test asd_mla_preprocess.
    Expectation: the result is correct
    """
    n = token_num
    head_num = 32
    hidden_strate = 7168
    block_num = block_num
    block_size = 128
    headdim = 576
    data_type = data_type
    cache_mode = 3
    mla_preprocess(n, head_num, hidden_strate, block_num, block_size, headdim, data_type, cache_mode, context_mode,
                   is_dyn=False)

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [ms.bfloat16])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('cache_mode', [0, 1, 2, 3])
def test_mla_preprocess_dynamic(data_type, cache_mode, context_mode):
    """
    Feature: test asd_mla_preprocess operator in graph mode
    Description: test asd_mla_preprocess.
    Expectation: the result is correct
    """
    n = 32
    head_num = 32
    hidden_strate = 7168
    block_num = 512
    block_size = 128
    headdim = 576
    data_type = data_type
    cache_mode = cache_mode
    mla_preprocess(n, head_num, hidden_strate, block_num, block_size, headdim, data_type, cache_mode, context_mode,
                   is_dyn=True)
