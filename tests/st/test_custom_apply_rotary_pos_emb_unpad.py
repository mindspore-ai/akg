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
import os
import numpy as np
import pytest

import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.np_dtype import bfloat16
from mindspore._c_expression import MSContext
import ms_custom_ops

def get_ms_dtype(query_dtype):
    if query_dtype == np.float32:
        ms_dtype = ms.float32
    elif query_dtype == np.float16:
        ms_dtype = ms.float16
    elif query_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype


class RotaryEmbedding(nn.Cell):
    # cosFormat=0  shape是[maxSeqLen, headDim]，    cos/sin不交替
    # cosFormat=1  shape是[maxSeqLen, headDim]，    cos/sin交替
    # cosFormat=2  shape是[batch*seqLen, headDim]， cos/sin不交替
    # cosFormat=3  shape是[batch*seqLen, headDim]， cos/sin交替
    def __init__(self, dim, base=10000, max_seq_len=2048, cos_dtype=np.float32, cos_format=0):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) * (1 / dim)))
        t = np.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = np.outer(t, inv_freq)
        if cos_format == 0 or cos_format == 2:
            emb = np.concatenate((freqs, freqs), axis=-1)
        else:
            freqs = np.expand_dims(freqs, 2)
            emb = np.concatenate((freqs, freqs), axis=-1)
            emb = emb.reshape(max_seq_len, dim)
        self.cos_np = np.cos(emb).astype(cos_dtype)
        self.sin_np = np.sin(emb).astype(cos_dtype)
        self.cos = Tensor(np.cos(emb), dtype=get_ms_dtype(cos_dtype))
        self.sin = Tensor(np.sin(emb), dtype=get_ms_dtype(cos_dtype))
        self.dim = dim
        self.cos_format = cos_format

    def construct(self, query, key, position_ids):
        query_embed, key_embed = ms_custom_ops.apply_rotary_pos_emb(query, key, self.cos, self.sin, position_ids, self.cos_format)
        return query_embed, key_embed

    def rope_compute(self, batch, headDim, hiddensize, hiddensizeQ, hiddensizeK, batchValidLen, headNum, headNumQ,headNumK, query, key, qtype):
        
        rotaryCoeff=2
        cos = self.cos.asnumpy()
        sin = self.sin.asnumpy()
        q_shape=query.shape
        k_shape=key.shape
        query=query.reshape(q_shape[0] * q_shape[1], q_shape[2])
        key=key.reshape(k_shape[0] * k_shape[1], k_shape[2])
        print("batch= {0}, headDim= {1}, hiddensize={2}, hiddensizeQ={3}, hiddensizeK={4},seqlen={5}, headNum={6}, headNumQ={7}, headNumK={8}, q={9}, k={10}, cos={11}, sin={12}".format(batch, headDim, hiddensize, hiddensizeQ, hiddensizeK, batchValidLen,headNum,headNumQ,headNumK,query.shape,key.shape,self.cos.shape,self.sin.shape))
        q = query.asnumpy()
        kk = key.asnumpy()
        seqlen = batchValidLen.asnumpy()
        ntokens = np.sum(seqlen)
        rope_q = np.zeros(shape=(ntokens, hiddensizeQ)).astype(qtype)
        rope_k = np.zeros(shape=(ntokens, hiddensizeK)).astype(qtype)
        prefix_Ntokens = 0
        cos_list = [cos[:x, :] for x in seqlen]
        sin_list = [sin[:x, :] for x in seqlen]
        cos=np.squeeze(np.concatenate(cos_list,axis=0))
        sin=np.squeeze(np.concatenate(sin_list,axis=0))
        cosTable = np.zeros(shape=(ntokens, hiddensize)).astype(qtype)
        for i in range(ntokens):
            for j in range(headNum):
                cosTable[i][j*headDim:(j+1)*headDim] = cos[i][:]
        for i in range(batch):
            curr_seqLen = seqlen[i]
            q1 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(qtype)
            k1 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(qtype)

            for i in range(prefix_Ntokens, prefix_Ntokens + curr_seqLen):
                q1[i-prefix_Ntokens] = q[i] * cosTable[i][:hiddensizeQ]
                k1[i-prefix_Ntokens] = kk[i] * cosTable[i][:hiddensizeK] 
            q2 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(qtype)
            k2 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(qtype)        
            for k in range(headNum):
                src_ = k * headDim
                dst_ = (k + 1) * headDim
                strdie = headDim // 2
                rotaryStrdie = headDim // rotaryCoeff
                rotaryTimesPerHead = rotaryCoeff / 2
                for cycle in range(int(rotaryTimesPerHead)):
                    src =  src_ + cycle * rotaryStrdie * 2
                    dst = src + rotaryStrdie * 2
                    for curr_seqLeni in range(curr_seqLen):
                        if k < headNumQ:
                            q2[curr_seqLeni][src:src + rotaryStrdie] = q[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                            q2[curr_seqLeni][src + rotaryStrdie:dst] = q[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                            q2[curr_seqLeni][src:dst] = q2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
                        if k < headNumK:
                            k2[curr_seqLeni][src:src + rotaryStrdie] = kk[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                            k2[curr_seqLeni][src + rotaryStrdie:dst] = kk[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                            k2[curr_seqLeni][src:dst] = k2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
            rope_q[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += q1 + q2
            rope_k[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += k1 + k2      
            
            prefix_Ntokens += curr_seqLen
        rope_q = rope_q.reshape(q_shape[0] , q_shape[1], q_shape[2])
        rope_k = rope_k.reshape(k_shape[0] , k_shape[1], k_shape[2])
        return rope_q, rope_k


def run(net, seqLens, num_head_q, num_head_k, hidden_dim, max_seq_len, query_dtype, pos_dtype):
    batch = len(seqLens)
    seqLen_= int(sum(seqLens)/2)
    hiddensizeQ = num_head_q * hidden_dim
    hiddensizeK = num_head_k * hidden_dim
    # query = np.random.rand(batch, seqLen, hiddensizeQ).astype(np.float32)
    # key = np.random.rand(batch, seqLen, hiddensizeK).astype(np.float32)
    query = np.random.rand(1, seqLen_, hiddensizeQ).astype(np.float32)
    key = np.random.rand(1, seqLen_, hiddensizeK).astype(np.float32)
    query=np.concatenate((query, query))
    key = np.concatenate((key, key))
    # 判断 q/k 前一半和后一半相等
    np.testing.assert_allclose(query[0:1, : , :], query[1:2, :, :], rtol=1e-2, atol=1e-2, err_msg="in query 前一半和后一半要相等")
    np.testing.assert_allclose(key[0:1, : , :], key[1:2, :, :], rtol=1e-2, atol=1e-2, err_msg="in key 前一半和后一半要相等")
    query=query.reshape(1, sum(seqLens), -1)
    key=key.reshape(1, sum(seqLens), -1)
    in_query = Tensor(query, dtype=get_ms_dtype(query_dtype))
    in_key = Tensor(key, dtype=get_ms_dtype(query_dtype))
    batch_valid_len = Tensor(seqLens, dtype=ms.int32)
    out_query, out_key = net(in_query, in_key, batch_valid_len)

    out_query_np = out_query.astype(ms.float32).asnumpy()
    out_key_np = out_key.astype(ms.float32).asnumpy()
    # np.testing.assert_allclose(out_query_np[0:1, : , :], out_query_np[1:2, :, :], rtol=1e-2, atol=1e-2, err_msg="out query 前一半和后一半要相等")
    # np.testing.assert_allclose(out_key_np[0:1, : , :], out_key_np[1:2, :, :], rtol=1e-2, atol=1e-2,  err_msg="out key 前一半和后一半要相等")
    np.testing.assert_allclose(out_query_np[:, :seqLen_ , :], out_query_np[:, seqLen_:, :], rtol=1e-2, atol=1e-2, err_msg="out query 前一半和后一半要相等")
    np.testing.assert_allclose(out_key_np[:, :seqLen_ , :], out_key_np[:, seqLen_:, :], rtol=1e-2, atol=1e-2,  err_msg="out key 前一半和后一半要相等")
    hiddensize = max(hiddensizeQ, hiddensizeK)
    headNum = max(num_head_q, num_head_k)
    golden_query, golden_key = net.rope_compute(batch, hidden_dim, hiddensize, hiddensizeQ, hiddensizeK, batch_valid_len, headNum, num_head_q, num_head_k, in_query, in_key, query_dtype)
    golden_query = golden_query.astype(np.float32)
    golden_key = golden_key.astype(np.float32)
    np.testing.assert_allclose(out_query_np, golden_query, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(out_key_np, golden_key, rtol=1e-2, atol=1e-2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('query_dtype', [np.float16])
@pytest.mark.parametrize('cos_dtype', [np.float16])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seq_len', [[4, 9, 4, 9]])
@pytest.mark.parametrize('num_head', [40])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rope_float16_unpad_special(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head, exec_mode):
    hidden_dim = 128
    base = 10000
    max_seq_len = 8192
    np.random.seed(0)
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})
    net = RotaryEmbedding(hidden_dim, base, max_seq_len, cos_dtype, cos_format)
    seqlens=np.array(seq_len, np.int32)
    run(net, seqlens, num_head, num_head, hidden_dim, max_seq_len, query_dtype, np.int32)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('query_dtype', [np.float16])
@pytest.mark.parametrize('cos_dtype', [np.float16, np.float32])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('batch_size', [2])
@pytest.mark.parametrize('seq_len', [[32,32], [1,1], [8192, 8192]])
@pytest.mark.parametrize('num_head', [8, 16])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rope_float16_unpad(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head, exec_mode):
    hidden_dim = 128
    base = 10000
    max_seq_len = 8192
    np.random.seed(0)
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})
    net = RotaryEmbedding(hidden_dim, base, max_seq_len, cos_dtype, cos_format)
    seqlens=np.array(seq_len, np.int32)
    run(net, seqlens, num_head, num_head, hidden_dim, max_seq_len, query_dtype, np.int32)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('query_dtype', [bfloat16])
@pytest.mark.parametrize('cos_dtype', [bfloat16])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('batch_size', [2])
@pytest.mark.parametrize('seq_len', [[32,32], [1,1], [8192, 8192]])
@pytest.mark.parametrize('num_head', [8, 16])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rope_float16_unpad_bf16(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head, exec_mode):
    hidden_dim = 128
    base = 10000
    max_seq_len = 8192
    np.random.seed(0)
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})
    net = RotaryEmbedding(hidden_dim, base, max_seq_len, cos_dtype, cos_format)
    seqlens=np.array(seq_len, np.int32)
    run(net, seqlens, num_head, num_head, hidden_dim, max_seq_len, query_dtype, np.int32)