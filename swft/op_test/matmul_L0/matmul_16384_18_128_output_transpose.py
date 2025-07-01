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

OP_NAME = 'matmul_16384_18_128_output_transpose'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

BATCH = 16384
M = 18
K = 128
N = 128

CORE_NUM = 2048
LOOP_SIZE = 8

# %1222 = "onnx.MatMul"(%pd_x_81, %253) {onnx_node_name = "9552_7197_1_train_fn_59043:CNode_58314:1248-9552_7197_1_train_fn_59043:CNode_58313:1247_1431"} : (tensor<16384x18x128xf32>, tensor<128x128xf32>) -> tensor<16384x18x128xf32>
# %1223 = "onnx.Reshape"(%1222, %7) {onnx_node_name = "9552_7197_1_train_fn_59043:CNode_58314:1248-9552_7197_1_train_fn_59043:CNode_58313:1247-9552_7197_1_train_fn_59043:CNode_58315:1249_339"} : (tensor<16384x18x128xf32>, tensor<4xi64>) -> tensor<16384x18x4x32xf32>
# %1224 = "onnx.Transpose"(%1223) {onnx_node_name = "9552_7197_1_train_fn_59043:CNode_58316:1250_1205", perm = [0, 2, 1, 3]} : (tensor<16384x18x4x32xf32>) -> tensor<16384x4x18x32xf32>


def gen_golden_data():
    gm_x = np.random.uniform(-0.3, 0.3, [BATCH, M, K]).astype(np.float32)
    gm_y = np.random.uniform(-0.3, 0.3, [K, N]).astype(np.float32)
    golden = np.matmul(gm_x.astype(np.float16), gm_y.astype(np.float16)).astype(
        np.float32).reshape(BATCH, M, 4, K // 4).transpose(0, 2, 1, 3)
    gm_x.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    gm_y.tofile(f"./temp/{OP_NAME}/input/gm_y.bin")
    golden.tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")


@sub_kernel(core_num=CORE_NUM)
def matmul_16384_18_128_output_transpose(gm_x, gm_y, gm_out):
    block_idx = get_block_idx()
    for i in range(LOOP_SIZE):
        ub_x = slice_to_ub(gm_x, [block_idx * (BATCH // CORE_NUM) * M + i * (
            BATCH // CORE_NUM // LOOP_SIZE) * M, 0], slicesize=[(BATCH // CORE_NUM // LOOP_SIZE) * M, K])
        ub_y = move_to_ub(gm_y)
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
        ub_out = change_view(
            ub_out, [BATCH // CORE_NUM // LOOP_SIZE, M, 4, N // 4])
        ub_out = transpose(ub_out, [0, 2, 1, 3])
        insert_to_gm(gm_out, ub_out, [block_idx * (BATCH // CORE_NUM) + i * (
            BATCH // CORE_NUM // LOOP_SIZE), 0, 0, 0], [BATCH // CORE_NUM // LOOP_SIZE, 4, M, N // 4])


if __name__ == "__main__":
    gen_golden_data()
    set_context("310P")
    gm_x = Tensor("GM", "FP32", [BATCH * M, K], format="ND", multi_core=False)
    gm_y = Tensor("GM", "FP32", [K, N], format="ND", multi_core=False)
    gm_out = Tensor("GM", "FP32", [BATCH, 4, M,
                    N // 4], format="ND", multi_core=False)
    compile_func(matmul_16384_18_128_output_transpose,
                 globals())(gm_x, gm_y, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['gm_x', 'gm_y'], outputs=['gm_out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin float32')
    sys.exit(return_code >> 8)