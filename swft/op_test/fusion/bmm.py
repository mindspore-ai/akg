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

OP_NAME = 'batch_matmul_nt'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

CORE_NUM = 8
TP = 4
BS = 1
B = 128 // TP
M = 1
K = 128
N = 512

# Numpy Test
def gen_golden_data():
    input0 = np.random.uniform(-1, 1, [BS, B, M, K]).astype(np.float16)
    input1 = np.random.uniform(-1, 1, [B, K, N]).astype(np.float16)
    input1_nz = input1.reshape((B, K, N//16, 16)).transpose(0, 2, 1, 3)
    golden = np.matmul(input0, input1).astype(np.float16)
    input0.tofile(f"./temp/{OP_NAME}/input/gm_a.bin")
    input1_nz.tofile(f"./temp/{OP_NAME}/input/gm_w.bin")
    golden.tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")
    print("golden shape: ", golden.shape)

# SWFT DSL
@sub_kernel(core_num=CORE_NUM)
def batch_matmul_nt(gm_a, gm_w, gm_out):
    idx = get_block_idx()
    l1_w = slice_to_l1(gm_w, [B // CORE_NUM * idx, 0, 0], [B // CORE_NUM, K, N])
    for i in range(BS):
        ub_a = slice_to_ub(gm_a, [i, B // CORE_NUM * idx, 0], [1, B // CORE_NUM, K])
        ub_a_view = change_view(ub_a, [B//CORE_NUM, M, K])
        ub_a_nz = nd_to_nz(ub_a_view)
        ub_a_pad = pad_to_ub(ub_a_nz, [B//CORE_NUM, 16, K])
        l1_a = move_to_l1(ub_a_pad)
        l0a = move_to_l0A(l1_a)
        for j in range(N // 64):
            l0b = slice_to_l0B(l1_w, [0, 0, j * 64], [B // CORE_NUM, K, 64])
            l0c = mmad(l0a, l0b)
            ub_c = move_to_ub(l0c, "FP16")
            ub_c_nz = slice_to_ub(ub_c, [0, 0, 0], [B // CORE_NUM, 1, 64])
            ub_c_nd = nz_to_nd(ub_c_nz)
            ub_c_nd = change_view(ub_c_nd, [1, B // CORE_NUM, 64])
            insert_to_gm(gm_out, ub_c_nd, [i, B // CORE_NUM * idx, 64 * j], [1, B // CORE_NUM, 64])

if __name__ == "__main__":
    gen_golden_data()
    set_context("310P")
    gm_a = Tensor("GM", "FP16", [BS, B, K], "ND", False)
    gm_w = Tensor("GM", "FP16", [B, K, N], "NZ", False)
    gm_out = Tensor("GM", "FP16", [BS, B, N], "ND", False)
    batch_matmul_nt(gm_a, gm_w, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME, hard_sync=True)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['gm_a', 'gm_w'], outputs=['gm_out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin float16')
    sys.exit(return_code >> 8)
