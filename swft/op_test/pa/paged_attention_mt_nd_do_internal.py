#!/usr/bin/env python3
# coding: utf-8
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

from swft.core import *
from swft.api import *
import numpy as np
import os
import sys
from random import randint

OP_NAME = "paged_attention_mt_16_128"

os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

SeqLen = 3
TP = 8
T = 45
B = 15
Hq = 128 // TP
D = 576
Skv = 512
BlockNum = 1024
BlockSize = 128
Hkv = 1
Dv = 512
Tor = 0.13523377
fixed_value = 250
MaxSkv = 32768
MaxBatch = 256

# Numpy Test
# ===============================================================================


def softmax(src):
    # 基于last轴进行rowmax(按行取最大值)处理
    max = np.max(src, axis=-1, keepdims=True)
    sub = src - max
    exp = np.exp(sub)
    # 基于last轴进行rowsum(按行求和)处理
    sum = np.sum(exp, axis=-1, keepdims=True)
    dst = exp / sum
    return dst


def gen_data():
    q_seq_len = np.random.randint(SeqLen, SeqLen + 1, [B], dtype=np.int32)
    global T
    T = np.sum(q_seq_len, 0, keepdims=False)
    context_length = np.full((B), fixed_value, dtype=np.int32)
    for i in range(B):
        context_length[i] = randint(q_seq_len[i], Skv)
    q = np.random.uniform(-1, 1, (T, Hq, D)).astype(np.float16)
    kv = np.random.uniform(-1, 1, (BlockNum, BlockSize,
                           Hkv, D)).astype(np.float16)
    kv_nz = kv.reshape(BlockNum, BlockSize, Hkv * D //
                       16, 16).transpose(0, 2, 1, 3)
    output = np.zeros((T, Hq, Dv)).astype(np.float16)
    block_table = np.zeros([B * Skv // BlockSize], dtype=np.int32)
    tot_length = 0
    seq_length = 0
    for i in range(B):
        length = (context_length[i] + BlockSize - 1) // BlockSize
        block_table[i*Skv // BlockSize:i*Skv // BlockSize+length] = np.random.choice(
            range(0, BlockNum, 1), length, replace=False).astype(np.int32)
        k_lst = []
        v_lst = []
        for j in range(length):
            x = block_table[i*Skv // BlockSize+j]
            if (j == length - 1 and context_length[i] % BlockSize != 0):
                tmp_k = kv[x, :context_length[i] % BlockSize, :, :].reshape(
                    context_length[i] % BlockSize, Hkv * D)
                k_lst.append(tmp_k)
                tmp_v = kv[x, :context_length[i] % BlockSize, :, :Dv].reshape(
                    context_length[i] % BlockSize, Hkv * Dv)
                v_lst.append(tmp_v)
            else:
                k_lst.append(kv[x, :, :, :].reshape(BlockSize, Hkv * D))
                v_lst.append(kv[x, :, :, :Dv].reshape(BlockSize, Hkv * Dv))
        k = np.concatenate(k_lst, 0)
        v = np.concatenate(v_lst, 0)
        query = q[seq_length:seq_length+q_seq_len[i], :, :].transpose(1, 0, 2)
        qk = np.matmul(query.reshape(
            Hq, q_seq_len[i], D).astype(np.float32), k.reshape(context_length[i], D).transpose(1, 0).astype(np.float32)).astype(np.float32)
        amask = np.ones(
            shape=[q_seq_len[i], context_length[i]]).astype(np.float32)
        amask = np.triu(
            amask, context_length[i] - q_seq_len[i] + 1) * (-10000.0)
        tmp = qk * Tor
        tmp += amask[:, :]
        o = softmax(tmp)
        output[seq_length:seq_length+q_seq_len[i], :, :] = np.matmul(o.astype(np.float16), v.reshape(
            context_length[i], Dv)).astype(np.float16).reshape(Hq, q_seq_len[i], Dv).transpose(1, 0, 2)
        tot_length += length
        seq_length += q_seq_len[i]
    output.astype(np.float16).tofile(
        f"./temp/{OP_NAME}/output/gm_pa_golden.bin")
    q.tofile(f"./temp/{OP_NAME}/input/gm_q.bin")
    kv_nz.tofile(f"./temp/{OP_NAME}/input/k_cache.bin")
    context_length.astype(np.int32).tofile(
        f"./temp/{OP_NAME}/input/context_length.bin")
    block_table.astype(np.int32).tofile(
        f"./temp/{OP_NAME}/input/block_table.bin")
    q_seq_len.astype(np.int32).tofile(f"./temp/{OP_NAME}/input/q_seq_len.bin")
    token_len = np.array([T], dtype=np.int32)
    token_len.tofile(f"./temp/{OP_NAME}/input/token_len.bin")
    skv = np.array([Skv // BlockSize], dtype=np.int32)
    skv.tofile(f"./temp/{OP_NAME}/input/skv.bin")
    tor = np.array([Tor], dtype=np.float32)
    tor.tofile(f"./temp/{OP_NAME}/input/tor.bin")
    batch_size = np.array([B], dtype=np.int32)
    batch_size.tofile(f"./temp/{OP_NAME}/input/batch_size.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=8)
def paged_attention_mt_16_128(gm_q, k_cache, block_table, context_length, q_seq_len, gm_pa, batch_size, skv, tor):
    block_idx = get_block_idx()
    ub_length = slice_to_ub(context_length, [0], [MaxBatch])
    ub_seq_len = slice_to_ub(q_seq_len, [0], [MaxBatch])
    mask = arange(0, 16, dtype="INT16")
    mask = vconv(mask, "FP16")
    ub_mask_lst = []
    for i in range(16):
        ub_mask_lst.append(mask)
    mask = concat(ub_mask_lst, 0)
    neg_inf = vector_dup(Scalar("FP16", -65500), [256], False)
    q_seq = Scalar("INT32", 0)
    now_core = Scalar("INT32", 0)
    for bs in dynamic_loop(batch_size):
        ub_table = slice_to_ub(
            block_table, [(bs) * skv], slicesize=[MaxSkv//BlockSize])
        seq_len = move_to_scalar(ub_seq_len[(bs)])
        for q_idx in dynamic_loop(seq_len):
            now_core.load(now_core + 1)
            if (now_core % 8 != block_idx):
                continue
            ub_q = slice_to_ub(
                gm_q, [q_seq + q_idx, Hq * D], slicesize=[1, Hq * D])
            ub_q = change_view(ub_q, [Hq, D], "ND")
            ub_q_nz = nd_to_nz(ub_q)
            l1_q = move_to_l1(ub_q_nz)
            length = move_to_scalar(ub_length[(bs)])
            block_num = (length + BlockSize - 1) // BlockSize
            gm = vector_dup(Scalar("FP16", -65500.0), [16, 1], False)
            gm = change_view(gm, new_format="NZ")
            gl = vector_dup(Scalar("FP32", 0), [16, 1], False)
            gl = change_view(gl, new_format="NZ")
            go = vector_dup(Scalar("FP32", 0.0), [16, D], False)
            go = change_view(go, new_format="NZ")
            tor = tor.astype("FP16")
            for j in dynamic_loop(block_num):
                x = move_to_scalar(ub_table[j])
                l1_k = slice_to_l1(
                    k_cache, [x, 0, 0], slicesize=[1, BlockSize, Hkv * D])
                l1_k = change_view(l1_k, [BlockSize, Hkv * D])
                for i in dynamic_loop(BlockSize // 16):
                    l0a = move_to_l0A(l1_q)
                    l0b = slice_to_l0B(
                        l1_k, [i * 16, 0], [16, Hkv * D], transpose=True)
                    l0c = mmad(l0a, l0b)
                    ub_qk = move_to_ub(l0c, "FP16")
                    ls = vmuls(ub_qk, tor)
                    ls = change_view(ls, new_shape=[256])
                    ub_mask = vcmpvs(
                        mask, (length - j * BlockSize - i * 16 - seq_len + q_idx + 1).astype("FP16"), 'LT')
                    ls = where(ls, neg_inf, ub_mask)
                    ls = change_view(ls, new_shape=[16, 16])
                    lm = vcmax(ls, -1)
                    hm = vmax(lm, gm)
                    dm = vsub(gm, hm)
                    dm = vconv(dm, "FP32")
                    dm = vexp(dm)
                    gm = move_to_ub(hm)
                    hm = vbrcb(hm, -1, ub_qk.shape[-1])
                    ls = vsub(ls, hm)
                    ls_f32 = vconv(ls, "FP32")
                    ls_f32 = vexp(ls_f32)
                    lp = vconv(ls_f32, "FP16")
                    ll = vcadd(ls_f32, -1)
                    gl = vmul(gl, dm)
                    gl = vadd(gl, ll)
                    l1_qk = move_to_l1(lp)
                    l0a = move_to_l0A(l1_qk)
                    l0b = slice_to_l0B(l1_k, [i * 16, 0], [16, Hkv * D])
                    l0c = mmad(l0a, l0b)
                    lo = move_to_ub(l0c, "FP32")
                    dm = vbrcb(dm, -1, go.shape[-1])
                    go = vmul(go, dm)
                    go = vadd(go, lo)
            gl = vbrcb(gl, -1, go.shape[-1])
            ub_out = vdiv(go, gl)
            ub_out = vconv(ub_out, "FP16")
            ub_out = nz_to_nd(ub_out)
            ub_out = slice_to_ub(ub_out, [0, 0], slicesize=[Hq, Dv])
            ub_out = change_view(ub_out, new_shape=[
                1, Hq * Dv], new_format="ND")
            insert_to_gm(gm_pa, ub_out, [q_seq + q_idx, Hq*Dv], slicesize=[1, Hq*Dv])
        q_seq.load(q_seq + seq_len)


if __name__ == "__main__":
    gen_data()
    set_context("310P")
    gm_q = Tensor("GM", "FP16", [T, Hq * D], format="ND", multi_core=False)
    k_cache = Tensor("GM", "FP16", [BlockNum, BlockSize, Hkv * D],
                     format="NZ", multi_core=False)
    block_table = Tensor(
        "GM", "INT32", [B * Skv // BlockSize], format="ND", multi_core=False)
    context_length = Tensor(
        "GM", "INT32", [B], format="ND", multi_core=False)
    q_seq_len = Tensor("GM", "INT32", [B], format="ND", multi_core=False)
    gm_pa = Tensor("GM", "FP16", [T, Hq * Dv],
                   format="ND", multi_core=False)
    batch_size = Scalar("INT32")
    skv = Scalar("INT32")
    tor = Scalar("FP32")
    compile_func(paged_attention_mt_16_128, globals())(
        gm_q, k_cache, block_table, context_length, q_seq_len, gm_pa, batch_size, skv, tor)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'gm_q', 'k_cache', 'context_length', 'block_table', 'q_seq_len', 'batch_size', 'skv', 'tor'], outputs=['gm_pa'], profiling=1000)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_pa_actual.bin ./temp/{OP_NAME}/output/gm_pa_golden.bin float16 4e-2 1e-2 4e-3')
    sys.exit(return_code >> 8)
