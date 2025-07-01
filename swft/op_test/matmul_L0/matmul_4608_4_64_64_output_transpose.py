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

OP_NAME = 'matmul_256_64_18_2'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

BATCH = 4608
BS = 4
M = 64
K = 64
N = 32

CORE_NUM = 1152
LOOP_SIZE = 4

# %448 = "onnx.MatMul"(%447, %394#2) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_266_CNode_57775:272_67"} : (tensor<4608x4x64x64xf32>, tensor<4608x4x64x32xf32>) -> tensor<4608x4x64x32xf32>
# %449 = "onnx.Transpose"(%448) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_236_CNode_57774:273_571", perm = [0, 2, 1, 3]} : (tensor<4608x4x64x32xf32>) -> tensor<4608x64x4x32xf32>
# %450 = "onnx.Reshape"(%449, %12) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_377_CNode_57772:275-9552_7197_1_train_fn_59043:equiv_246_input_ms:276_585"} : (tensor<4608x64x4x32xf32>, tensor<2xi64>) -> tensor<294912x128xf32>

# Numpy Test
# ===============================================================================


def gen_golden_data():
    gm_x = np.random.uniform(-0.3, 0.3, [BATCH, BS, M, K]).astype(np.float32)
    gm_y = np.random.uniform(-0.3, 0.3, [BATCH, BS, K, N]).astype(np.float32)
    matmul_res = np.matmul(gm_x.astype(np.float16),
                           gm_y.astype(np.float16)).astype(np.float32)
    matmul_res = matmul_res.transpose(0, 2, 1, 3).reshape((294912, 128))
    gm_x.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    gm_y.tofile(f"./temp/{OP_NAME}/input/gm_y.bin")
    matmul_res.tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=CORE_NUM)
def matmul_4608x4x64x64_output_transpose(gm_x, gm_y, gm_out):
    block_idx = get_block_idx()
    for i in range(LOOP_SIZE):
        # for j in range(LOOP_SIZE):
        ub_x = slice_to_ub(gm_x, [block_idx * (BATCH * BS // CORE_NUM) + i * (BATCH * BS //
                           CORE_NUM // LOOP_SIZE), 0, 0], slicesize=[(BATCH * BS // CORE_NUM // LOOP_SIZE), M, K])
        ub_y = slice_to_ub(gm_y, [block_idx * (BATCH * BS // CORE_NUM) + i * (BATCH * BS //
                           CORE_NUM // LOOP_SIZE), 0, 0], slicesize=[(BATCH * BS // CORE_NUM // LOOP_SIZE), K, N])
        ub_x_half = vconv(ub_x, "FP16")
        ub_y_half = vconv(ub_y, "FP16")
        ub_x_nz = nd_to_nz(ub_x_half)
        ub_y_nz = nd_to_nz(ub_y_half)
        l1_x = move_to_l1(ub_x_nz)
        l1_y = move_to_l1(ub_y_nz)
        l0a = move_to_l0A(l1_x)
        l0b = move_to_l0B(l1_y)
        l0c = mmad(l0a, l0b)
        ub_out_nz = move_to_ub(l0c)
        ub_out = nz_to_nd(ub_out_nz)
        ub_out = change_view(ub_out, [1, 4, 64, 32])
        ub_out = transpose(ub_out, [0, 2, 1, 3])
        insert_to_gm(gm_out, ub_out, [block_idx * (BATCH // CORE_NUM) +
                     i * (BATCH // CORE_NUM // LOOP_SIZE), 0, 0, 0], [1, M, BS, N])


if __name__ == "__main__":
    set_context("310P")
    gen_golden_data()
    gm_x = Tensor("GM", "FP32", [BATCH * BS, M, K], "ND", False)
    gm_y = Tensor("GM", "FP32", [BATCH * BS, K, N], "ND", False)
    gm_out = Tensor("GM", "FP32", [BATCH, M, BS, N], "ND", False)
    compile_func(matmul_4608x4x64x64_output_transpose,
                 globals())(gm_x, gm_y, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME, hard_sync=True)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['gm_x', 'gm_y'], outputs=['gm_out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin float32')
    sys.exit(return_code >> 8)
