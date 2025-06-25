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

import numpy as np
import os
import sys
from swft.core import *
from swft.api import *
TP = 4
NUM_TOKENS = 1
CORE_NUM = 8
OP_NAME = 'preMLA'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

# Numpy Test
# ===============================================================================


def rms_norm(input, gamma):
    input_sum = np.sqrt(np.sum((input*input).astype("float32"), -1,
                        keepdims=True) / input.shape[-1]).astype("float16")
    return input/input_sum*gamma


def rope(input, rope_sin, rope_cos):
    input_first = input[:, :, :, :32]
    input_second = input[:, :, :, 32:]
    input_RQ = np.concatenate(
        (input_second * (-1), input_first), -1).reshape(input.shape)
    return input * rope_cos + input_RQ * rope_sin


def gen_golden_data():
    x = np.random.uniform(-1, 1, [1, NUM_TOKENS, 7168]).astype(np.float16)
    wqa = np.random.uniform(-1, 1, [7168, 1536]).astype(np.float16)
    wqa_nz = wqa.reshape((7168, 1536//16, 16)).transpose(1, 0, 2)
    wqb = np.random.uniform(-0.3, 0.3, [1536, 128//TP*192]).astype(np.float16)
    wqb_nz = wqb.reshape((1536, 128//TP*192//16, 16)).transpose(1, 0, 2)
    wkv_a = np.random.uniform(-1, 1, (7168, 576)).astype("float16")
    wkv_a_nz = wkv_a.reshape((7168, 576//16, 16)).transpose(1, 0, 2)
    rmsNormkv_gamma = np.random.uniform(-1, 1, (512)).astype("float16")
    wkv_b_1 = np.random.uniform(-0.3, 0.3,
                                (128//TP, 128, 512)).astype("float16")
    wkv_b_1_nz = wkv_b_1.reshape(
        (128//TP, 128, 512//16, 16)).transpose(0, 2, 1, 3)
    rmsNormq_gamma = np.random.uniform(-1, 1, (1536,)).astype("float16")
    rope_sin = np.random.uniform(-1, 1,
                                 (1, 1, NUM_TOKENS, 64,)).astype("float16")
    rope_cos = np.random.uniform(-1, 1,
                                 (1, 1, NUM_TOKENS, 64,)).astype("float16")
    sync = np.zeros(8).astype(np.int32)
    res = np.matmul(x, wqa).astype(np.float16)
    bslq = rms_norm(res, rmsNormq_gamma)
    # BS, 1, 4, 192
    BSH = np.matmul(bslq, wqb).astype(
        np.float16).reshape((1, NUM_TOKENS, 128//TP, 192))
    BSND = BSH[:, :, :, :128]
    BSND_r = BSH[:, :, :, 128:]
    BSND_r = rope(BSND_r, rope_sin, rope_cos)
    BSND = BSND.reshape((-1, 1, 128))
    wkv_b_1 = wkv_b_1.reshape((-1, 128, 512))
    BSNC = np.matmul(BSND.reshape(1, NUM_TOKENS, 128//TP, 1,
                     128).astype(np.float16), wkv_b_1.astype(np.float16))
    BSNC = BSNC.reshape((1, NUM_TOKENS, 128//TP, -1)).astype(np.float16)
    q = np.concatenate((BSNC, BSND_r), axis=-1)

    # kv
    BSL_kvr = np.dot(x, wkv_a)
    BSC = BSL_kvr[:, :, :512]
    BSL_r = BSL_kvr[:, :, 512:]
    BSC = rms_norm(BSC, rmsNormkv_gamma)
    BS1L_r = BSL_r.reshape((1, 1, 1, 64))
    BS1L_r = rope(BS1L_r, rope_sin, rope_cos)
    BSL_r = BS1L_r.reshape(BSL_r.shape)
    kv = np.concatenate((BSC, BSL_r), axis=-1)
    q = q.reshape((1, 128//TP, -1)).transpose(1, 0, 2)
    q_pad = np.pad(q, ((0, 0), (0, 16 - 1), (0, 0)), 'constant')
    q_nz = q_pad.reshape(128//TP, 16, 576//16, 16).transpose(0, 2, 1, 3)
    x.tofile(f"./temp/{OP_NAME}/input/x.bin")
    wqa_nz.tofile(f"./temp/{OP_NAME}/input/wqa.bin")
    rmsNormq_gamma.tofile(f"./temp/{OP_NAME}/input/rmsNormq_gamma.bin")
    rmsNormkv_gamma.tofile(f"./temp/{OP_NAME}/input/rmsNormkv_gamma.bin")
    rope_sin.tofile(f"./temp/{OP_NAME}/input/rope_sin.bin")
    rope_cos.tofile(f"./temp/{OP_NAME}/input/rope_cos.bin")
    wqb_nz.tofile(f"./temp/{OP_NAME}/input/wqb.bin")
    wkv_a_nz.tofile(f"./temp/{OP_NAME}/input/wkv_a.bin")
    wkv_b_1_nz.tofile(f"./temp/{OP_NAME}/input/wkv_b_1.bin")
    q_nz.tofile(f"./temp/{OP_NAME}/output/Q_golden.bin")
    kv.tofile(f"./temp/{OP_NAME}//output/KV_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=CORE_NUM)
def matmul_1_7168_1536(x, wqa, temp1_gm):
    K = 7168
    block_idx = get_block_idx()
    l1_tiling = 4
    l0_tiling = 14
    l1_size = 7168 // CORE_NUM // l1_tiling
    l0_size = l1_size // l0_tiling
    for k1 in range(l1_tiling):
        ub_x = slice_to_ub(
            x, [0, 0, block_idx * 896 + k1 * l1_size], [1, 1, l1_size])
        ub_x_view = change_view(ub_x, new_shape=[1, l1_size], new_format="NZ")
        ub_x_pad = pad_to_ub(ub_x_view, [16, l1_size])
        l1_x = move_to_l1(ub_x_pad)
        l1_wqa = slice_to_l1(
            wqa, [block_idx * 896 + k1 * l1_size, 0], [l1_size, 1536])
        for k2 in range(l0_tiling):
            l0_a = slice_to_l0A(l1_x, [0, k2 * l0_size], [16, l0_size])
            l0_b = slice_to_l0B(l1_wqa, [k2 * l0_size, 0], [l0_size, 1536])
            if k1 == 0 and k2 == 0:
                l0_c = mmad(l0_a, l0_b)
            else:
                l0_c = mmad(l0_a, l0_b, l0_c)
    ub_out = move_to_ub(l0_c, "FP16")
    ub_out = change_view(ub_out, new_shape=[
                         1, ub_out.shape[0], ub_out.shape[1]])
    ub_slice = slice_to_ub(ub_out, [0, 0, 0], [1, 1, 1536])
    insert_to_gm(temp1_gm, ub_slice, [block_idx, 0, 0], [1, 1, 1536])


@sub_kernel(core_num=1)
def reduce_sum_8_1536_rms_norm(temp1_gm, rmsNormq_gamma, BSL_q):
    ub_gamma = move_to_ub(rmsNormq_gamma)
    for j in range(CORE_NUM):
        ub_x = slice_to_ub(temp1_gm, [j, 0, 0], [1, 1, 1536])
        ub_x_32 = vconv(ub_x, "FP32")
        if j == 0:
            ub_out = ub_x_32
        else:
            ub_out = vadd(ub_out, ub_x_32)
    ub_out_square = vmul(ub_out, ub_out)
    ub_reduce_add = vcadd(ub_out_square, reduce_axis=-1)
    scalar_sum = (1536 / move_to_scalar(ub_reduce_add[0])).sqrt()
    ub_rmsnorm = vmuls(ub_out, scalar_sum)
    ub_rmsnorm_f16 = vconv(ub_rmsnorm, "FP16")
    ub_rmsnorm_f16 = vmul(ub_rmsnorm_f16, ub_gamma)
    BSL_q.load(ub_rmsnorm_f16)


@sub_kernel(core_num=CORE_NUM)
def matmul_1_1536_32_192(BSL_q, wqb, rope_sin, rope_cos, wkv_b_1, Q):
    N = 6144
    block_idx = get_block_idx()
    l1_tiling = 3
    l0_tiling = 12
    l1_size = N // CORE_NUM // l1_tiling
    ub_x = move_to_ub(BSL_q)
    ub_x_pad = pad_to_ub(ub_x, [1, 16, 1536])
    ub_x_pad = change_view(ub_x_pad, new_shape=[16, 1536])
    l1_x_pad = move_to_l1(ub_x_pad)
    ub_outs = []
    for n in range(l1_tiling):
        # 1536, 768
        l1_wqb = slice_to_l1(
            wqb, [0, block_idx * (N // CORE_NUM) + n * l1_size], [1536, l1_size])
        for k in range(l0_tiling):
            l0_a = slice_to_l0A(l1_x_pad, [0, k * 128], [16, 128])
            l0_b = slice_to_l0B(l1_wqb, [k * 128, 0], [128, l1_size])
            if k == 0:
                l0_c = mmad(l0_a, l0_b)
            else:
                l0_c = mmad(l0_a, l0_b, l0_c)
        ub_outs.append(move_to_ub(l0_c, "FP16"))
    ub_out = concat(ub_outs, 1)
    ub_out = change_view(ub_out, new_shape=[
                         1, ub_out.shape[0], ub_out.shape[1]])
    ub_slice = slice_to_ub(ub_out, [0, 0, 0], [1, 1, 768])
    ub_BSH = change_view(ub_slice, new_shape=[4, 192], new_format="ND")
    ub_sin = move_to_ub(rope_sin)
    ub_cos = move_to_ub(rope_cos)
    ub_BSND = slice_to_ub(ub_BSH, [0, 0], [4, 128])
    ub_BSND_r = slice_to_ub(ub_BSH, [0, 128], [4, 64])
    ub_BSND_r_0 = slice_to_ub(ub_BSND_r, [0, 0], [4, 32])
    ub_BSND_r_1 = slice_to_ub(ub_BSND_r, [0, 32], [4, 32])
    ub_cos_brcb = vbrcb(ub_cos, broadcast_axis=0, broad_size=4)
    ub_sin_brcb = vbrcb(ub_sin, broadcast_axis=0, broad_size=4)
    ub_adder_0 = vmul(ub_BSND_r, ub_cos_brcb)
    ub_adder_1 = vmul(
        concat([vmuls(ub_BSND_r_1, -1), ub_BSND_r_0], 1), ub_sin_brcb)
    ub_rope = vadd(ub_adder_0, ub_adder_1)  # 4, 64
    # matmul 4,128 @ 128,512
    l1_wkv_b_1 = slice_to_l1(wkv_b_1, [block_idx * 4, 0, 0], [4, 128, 512])
    ub_x = change_view(ub_BSND, [4, 1, 128])
    ub_x_pad = pad_to_ub(ub_x, [4, 16, 128])
    ub_x_pad = nd_to_nz(ub_x_pad)
    l1_x_pad = move_to_l1(ub_x_pad)
    l0_a = move_to_l0A(l1_x_pad)
    for k in range(512 // 16):
        l0_b = slice_to_l0B(l1_wkv_b_1, [0, 0, k * 16], [4, 128, 16])
        l0_c = mmad(l0_a, l0_b)
        ub_out_pad = move_to_ub(l0_c, "FP16")
        ub_BSNC = slice_to_ub(ub_out_pad, [0, 0, 0], [4, 1, 16])
        insert_to_gm(Q, ub_BSNC, [block_idx * 4, 0, k * 16], [4, 1, 16])
    ub_rope = change_view(ub_rope, [4, 1, 64])
    ub_rope = nd_to_nz(ub_rope)
    insert_to_gm(Q, ub_rope, [block_idx * 4, 0, 512], [4, 1, 64])


@sub_kernel(core_num=CORE_NUM)
def matmul_1_7168_576(x, wkv_a, temp2_gm):
    K = 7168
    block_idx = get_block_idx()
    l1_tiling = 2
    l0_tiling = 14
    l1_size = 7168 // CORE_NUM // l1_tiling
    l0_size = l1_size // l0_tiling
    for k1 in range(l1_tiling):
        ub_x = slice_to_ub(
            x, [0, 0, block_idx * 896 + k1 * l1_size], [1, 1, l1_size])
        ub_x_view = change_view(ub_x, new_shape=[1, l1_size], new_format="NZ")
        ub_x_pad = pad_to_ub(ub_x_view, [16, l1_size])
        l1_x = move_to_l1(ub_x_pad)
        l1_wkv_a = slice_to_l1(
            wkv_a, [block_idx * 896 + k1 * l1_size, 0], [l1_size, 576])
        for k2 in range(l0_tiling):
            l0_a = slice_to_l0A(l1_x, [0, k2 * l0_size], [16, l0_size])
            l0_b = slice_to_l0B(l1_wkv_a, [k2 * l0_size, 0], [l0_size, 576])
            if k1 == 0 and k2 == 0:
                l0_c = mmad(l0_a, l0_b)
            else:
                l0_c = mmad(l0_a, l0_b, l0_c)
    ub_out = move_to_ub(l0_c, "FP16")
    ub_out = change_view(ub_out, new_shape=[
                         1, ub_out.shape[0], ub_out.shape[1]])
    ub_slice = slice_to_ub(ub_out, [0, 0, 0], [1, 1, 576])
    insert_to_gm(temp2_gm, ub_slice, [block_idx, 0, 0], [1, 1, 576])


@sub_kernel(core_num=1)
def reduce_sum_8_576_rms_norm_rope_concat(temp2_gm, rmsNormkv_gamma, rope_sin, rope_cos, kv):
    ub_gamma = move_to_ub(rmsNormkv_gamma)
    for j in range(CORE_NUM):
        ub_x = slice_to_ub(temp2_gm, [j, 0, 0], [1, 1, 576])
        ub_x_32 = vconv(ub_x, "FP32")
        if j == 0:
            ub_out = ub_x_32
        else:
            ub_out = vadd(ub_out, ub_x_32)
    ub_out = change_view(ub_out, new_format="ND", new_shape=[1, 576])
    ub_bsc = slice_to_ub(ub_out, [0, 0], [1, 512])
    ub_bs1l_r = vconv(slice_to_ub(ub_out, [0, 512], [1, 64]), "FP16")
    # rmsnorm
    ub_out_square = vmul(ub_bsc, ub_bsc)
    ub_reduce_add = vcadd(ub_out_square, reduce_axis=-1)
    scalar_sum = (512 / move_to_scalar(ub_reduce_add[0])).sqrt()
    ub_rmsnorm = vmuls(ub_bsc, scalar_sum)
    ub_rmsnorm_f16 = vconv(ub_rmsnorm, "FP16")
    ub_rmsnorm_f16 = vmul(ub_rmsnorm_f16, ub_gamma)
    # rope
    ub_sin = move_to_ub(rope_sin)
    ub_cos = move_to_ub(rope_cos)
    ub_bs1l_r = change_view(ub_bs1l_r, new_format="ND", new_shape=[1, 64])
    ub_bs1l_r_0 = slice_to_ub(ub_bs1l_r, [0, 0], [1, 32])
    ub_bs1l_r_1 = slice_to_ub(ub_bs1l_r, [0, 32], [1, 32])
    ub_adder_0 = vmul(ub_bs1l_r, ub_cos)
    ub_adder_1 = vmul(concat([vmuls(ub_bs1l_r_1, -1), ub_bs1l_r_0], 1), ub_sin)
    ub_rope = vadd(ub_adder_0, ub_adder_1)
    ub_kv = concat([ub_rmsnorm_f16, ub_rope], 1)
    ub_kv = kv.load(ub_kv)


if __name__ == '__main__':
    gen_golden_data()
    set_context("310P")
    x = Tensor("GM", "FP16", [1, NUM_TOKENS, 7168],
               format="ND", multi_core=False)
    wqa = Tensor("GM", "FP16", [7168, 1536], format="NZ", multi_core=False)
    sync1 = Tensor("GM", "INT32", [64], format="ND", multi_core=False)
    sync2 = Tensor("GM", "INT32", [64], format="ND", multi_core=False)
    sync3 = Tensor("GM", "INT32", [64], format="ND", multi_core=False)
    temp1_gm = Tensor(
        "GM", "FP16", [CORE_NUM, 1, 1536], format="NZ", multi_core=False)
    BSL_q = Tensor("GM", "FP16", [1, NUM_TOKENS,
                   1536], format="NZ", multi_core=False)
    rmsNormq_gamma = Tensor(
        "GM", "FP16", [1536], format="ND", multi_core=False)
    matmul_1_7168_1536(x, wqa, temp1_gm)
    sync_cores(sync1)
    reduce_sum_8_1536_rms_norm(temp1_gm, rmsNormq_gamma, BSL_q)
    wqb = Tensor("GM", "FP16", [1536, 128//TP*192],
                 format="NZ", multi_core=False)
    BSH = Tensor("GM", "FP16", [1, NUM_TOKENS, 128 //
                 TP*192], format="NZ", multi_core=False)
    sync_cores(sync2)
    wkv_b_1 = Tensor(
        "GM", "FP16", [128//TP, 128, 512], format="NZ", multi_core=False)
    Q = Tensor("GM", "FP16", [128//TP, 16, 576], format="NZ", multi_core=False)
    rope_sin = Tensor(
        "GM", "FP16", [NUM_TOKENS, 64], format="ND", multi_core=False)
    rope_cos = Tensor(
        "GM", "FP16", [NUM_TOKENS, 64], format="ND", multi_core=False)
    matmul_1_1536_32_192(BSL_q, wqb, rope_sin, rope_cos, wkv_b_1, Q)
    temp2_gm = Tensor(
        "GM", "FP16", [CORE_NUM, 1, 576], format="NZ", multi_core=False)
    wkv_a = Tensor("GM", "FP16", [7168, 576], format="NZ", multi_core=False)
    KV = Tensor("GM", "FP16", [1, 576], format="ND", multi_core=False)
    matmul_1_7168_576(x, wkv_a, temp2_gm)
    rmsNormkv_gamma = Tensor(
        "GM", "FP16", [512], format="ND", multi_core=False)
    sync_cores(sync3)
    reduce_sum_8_576_rms_norm_rope_concat(
        temp2_gm, rmsNormkv_gamma, rope_sin, rope_cos, KV)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['x', 'wqa', 'rmsNormq_gamma', 'wqb', 'rope_sin',
                'rope_cos', 'wkv_b_1', 'wkv_a', 'rmsNormkv_gamma'], outputs=['Q', 'KV'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for output in ['Q', 'KV']:
        return_code = os.system(
            f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/{output}_actual.bin ./temp/{OP_NAME}/output/{output}_golden.bin')
        if return_code != 0:
            sys.exit(return_code >> 8)
