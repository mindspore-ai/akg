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

OP_NAME = "paged_attention_tp8"

os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

SeqLen = 128
TP = 8
B = 3
S = 1
Hq = 128 // TP
D = 576
Skv = 4096
BlockNum = 512
BlockSize = 16
Hkv = 1
Dv = 512
tor = 0.13523377
fixed_value = 32

# Numpy Test
# ===============================================================================


def softmax(src):
    #基于last轴进行rowmax(按行取最大值)处理
    max = np.max(src, axis=-1, keepdims=True)
    sub = src - max
    exp = np.exp(sub)
    #基于last轴进行rowsum(按行求和)处理
    sum = np.sum(exp, axis=-1, keepdims=True)
    dst = exp / sum
    return dst

def gen_data():
    q = np.random.uniform(-1, 1, (B, S, Hq, D)).astype(np.float16)
    round_bs = (B*S + 15) // 16 * 16
    q_nz = np.pad(q.reshape(B * S, Hq * D), ((0, round_bs - B * S), (0, 0)), "constant")
    q_nz = q_nz.reshape(16, Hq * D // 16, 16).transpose(1, 0, 2)
    kv = np.random.uniform(-1, 1, (BlockNum, BlockSize, Hkv, D)).astype(np.float16)
    kv_nz = kv.reshape(BlockNum, BlockSize, Hkv * D // 16, 16).transpose(0, 2, 1, 3)
    output = np.zeros((B, S, Hq, Dv)).astype(np.float16)
    block_table = np.zeros([B, Skv // BlockSize], dtype=np.int32)
    context_length = np.full((B), fixed_value, dtype=np.int32) * BlockSize
    tot_length = 0
    for i in range(len(context_length)):
        length = (context_length[i] + BlockSize - 1) // BlockSize
        block_table[i][:length] = np.random.choice(range(0, BlockNum, 1), length, replace=False).astype(np.int32)
        k_lst = []
        v_lst = []
        for j in range(length):
            x = block_table[i][j]
            if (j == length - 1 and context_length[i] % BlockSize != 0):
                tmp_k = kv[x, :context_length[i] % BlockSize, :, :].reshape(context_length[i] % BlockSize, Hkv * D)
                k_lst.append(tmp_k)
                tmp_v = kv[x, :context_length[i] % BlockSize, :, :Dv].reshape(context_length[i] % BlockSize, Hkv * Dv)
                v_lst.append(tmp_v)
            else:
                k_lst.append(kv[x, :, :, :].reshape(BlockSize, Hkv * D))
                v_lst.append(kv[x, :, :, :Dv].reshape(BlockSize, Hkv * Dv))
        k = np.concatenate(k_lst, 0)
        v = np.concatenate(v_lst, 0)
        qk = np.matmul(q[i].reshape(S * Hq, D), k.reshape(context_length[i], D).transpose(1, 0)).astype(np.float32)
        o = softmax(qk * tor)
        output[i] = np.matmul(o.astype(np.float16), v.reshape(context_length[i], Dv)).astype(np.float16).reshape(S, Hq, Dv)
        tot_length += length
    output_nz = np.pad(output.reshape(B * S, Hq * Dv), ((0, round_bs - B * S), (0, 0)), "constant")
    output_nz = output_nz.reshape(round_bs, Hq * Dv // 16, 16).transpose(1, 0, 2)
    output_nz.astype(np.float16).tofile(f"./temp/{OP_NAME}/output/gm_pa_golden.bin")
    q_nz.tofile(f"./temp/{OP_NAME}/input/gm_q.bin")
    kv_nz.tofile(f"./temp/{OP_NAME}/input/k_cache.bin")
    context_length.astype(np.int32).tofile(f"./temp/{OP_NAME}/input/context_length.bin")
    block_table.astype(np.int32).tofile(f"./temp/{OP_NAME}/input/block_table.bin")
    batch_size = np.array([B], dtype=np.int32)
    batch_size.tofile(f"./temp/{OP_NAME}/input/batch_size.bin")


# OP Impl
# ===============================================================================


@sub_kernel(core_num=8)
def paged_attention_tp8(gm_q, k_cache, block_table, context_length, gm_pa, batch_size):
    block_idx = get_block_idx()
    ub_length = move_to_ub(context_length)
    for bs in dynamic_loop(batch_size):
        ub_table = slice_to_ub(block_table, [bs, 0], slicesize=[1, Skv//BlockSize])
        ub_table = change_view(ub_table, new_shape=[Skv//BlockSize])
        ub_q = slice_to_ub(gm_q, [bs, block_idx * (Hq // 8) * D], slicesize=[S, Hq // 8 * D])
        ub_q = change_view(ub_q, [Hq // 8, D], "ND")
        if (TP != 1):
            ub_q = pad_to_ub(ub_q, [16, D])
        ub_q_nz = nd_to_nz(ub_q)
        l1_q = move_to_l1(ub_q_nz)
        length = move_to_scalar(ub_length[bs])
        length = (length + BlockSize - 1) // BlockSize
        gm = vector_dup(Scalar("FP16", -65500.0), [1, 16, 1], False)
        gm = change_view(gm, new_format="NZ")
        gl = vector_dup(Scalar("FP32", 0), [1, 16, 1], False)
        gl = change_view(gl, new_format="NZ")
        go = vector_dup(Scalar("FP32", 0.0), [1, 16, D], False)
        go = change_view(go, new_format="NZ")
        for j in dynamic_loop(length):
            x = move_to_scalar(ub_table[j])
            for i in range(BlockSize // 16):
                y = i % (BlockSize // 16)
                l1_k = slice_to_l1(
                    k_cache, [x, y * 16, 0], slicesize=[1, 16, Hkv * D])    # [1, 16, 576]
                l1_v = slice_to_l1(
                    k_cache, [x, y * 16, 0], slicesize=[1, 16, Hkv * D])    # [1, 16, 576]
                l0a = move_to_l0A(l1_q)                                     # [1, 16, 576]
                l0b = move_to_l0B(l1_k, Transpose=True)                     # [1, 576, 16]
                l0c = mmad(l0a, l0b)                                        # [1, 16, 16]
                ub_qk = move_to_ub(l0c, "FP16")                             # [1, 16, 16]
                ls = vmuls(ub_qk, tor)                                      # [1, 16, 16]
                lm = vcmax(ls, -1)                                          # [1, 16, 1]
                hm = vmax(lm, gm)                                           # [1, 16, 1]
                dm = vsub(gm, hm)                                           # [1, 16, 1]
                dm = vconv(dm, "FP32")                                      # [1, 16, 1]
                dm = vexp(dm)                                               # [1, 16, 1]
                gm = move_to_ub(hm)                                         # [1, 16, 1]
                hm = vbrcb(hm, -1, ub_qk.shape[-1])                         # [1, 16, 16]
                ls = vsub(ls, hm)                                           # [1, 16, 16]
                ls_f32 = vconv(ls, "FP32")                                  # [1, 16, 16]
                ls_f32 = vexp(ls_f32)                                       # [1, 16, 16]
                lp = vconv(ls_f32, "FP16")                                  # [1, 16, 16]
                ll = vcadd(ls_f32, -1)                                      # [1, 16, 1]
                gl = vmul(gl, dm)                                           # [1, 16, 1]
                gl = vadd(gl, ll)                                           # [1, 16, 1]
                l1_qk = move_to_l1(lp)                                      # [1, 16, 16]
                l0a = move_to_l0A(l1_qk)                                    # [1, 16, 16]
                l0b = move_to_l0B(l1_v)                                     # [1, 16, 576]
                l0c = mmad(l0a, l0b)                                        # [1, 16, 576]
                lo = move_to_ub(l0c, "FP32")                                # [1, 16, 576]
                dm = vbrcb(dm, -1, go.shape[-1])
                go = vmul(go, dm)
                go = vadd(go, lo)
        gl = vbrcb(gl, -1, go.shape[-1])
        ub_out = vdiv(go, gl)
        ub_out = vconv(ub_out, "FP16")
        ub_out = nz_to_nd(ub_out)
        ub_out = slice_to_ub(ub_out, [0, 0, 0], slicesize=[1, Hq//8, Dv])
        ub_out = change_view(ub_out, new_shape=[S, Hq//8 * Dv], new_format="NZ")
        insert_to_gm(gm_pa, ub_out, [bs, block_idx *
                    (Hq//8) * Dv], slicesize=[S, Hq//8*Dv])



if __name__ == "__main__":
    gen_data()
    set_context("310P")
    gm_q = Tensor("GM", "FP16", [16, Hq * D], format="NZ", multi_core=False)
    k_cache = Tensor("GM", "FP16", [BlockNum, BlockSize, Hkv * D],
                     format="NZ", multi_core=False)
    block_table = Tensor(
        "GM", "INT32", [B, Skv // BlockSize], format="ND", multi_core=False)
    context_length = Tensor(
        "GM", "INT32", [B], format="ND", multi_core=False)
    gm_pa = Tensor("GM", "FP16", [16, Hq * Dv],
                   format="NZ", multi_core=False)
    batch_size = Scalar("INT32")
    paged_attention_tp8(gm_q, k_cache, block_table, context_length, gm_pa, batch_size)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'gm_q', 'k_cache', 'context_length', 'block_table', 'batch_size'], outputs=['gm_pa'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_pa_actual.bin ./temp/{OP_NAME}/output/gm_pa_golden.bin float16 4e-2 1e-2 4e-3')
    sys.exit(return_code >> 8)

