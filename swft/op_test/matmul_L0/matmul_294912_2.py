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

OP_NAME = 'matmul_294912_2'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

M = 2
K = 294912
N = 128

CORE_NUM = 128
LOOP_SIZE = 16

# %976 = "onnx.Transpose"(%649) {onnx_node_name = "onnx.Transpose_180", perm = [1, 0]} : (tensor<294912x2xf32>) -> tensor<2x294912xf32>
# %977 = "onnx.MatMul"(%976, %567) {onnx_node_name = "9552_7197_1_train_fn_59043:CNode_58245:945_183"} : (tensor<2x294912xf32>, tensor<294912x128xf32>) -> tensor<2x128xf32>


def gen_golden_data():
    gm_x = np.random.uniform(-0.3, 0.3, [K, M]).astype(np.float32)
    gm_y = np.random.uniform(-0.3, 0.3, [K, N]).astype(np.float32)
    golden = np.matmul(gm_x.transpose(1, 0).astype(
        np.float16), gm_y.astype(np.float16)).astype(np.float32)
    gm_x.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    gm_y.tofile(f"./temp/{OP_NAME}/input/gm_y.bin")
    golden.tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")


@sub_kernel(core_num=CORE_NUM)
def matmul_294912_2(gm_x, gm_y, gm_out):
    block_idx = get_block_idx()
    for i in range(LOOP_SIZE):
        ub_x = slice_to_ub(gm_x, [block_idx * (K // CORE_NUM) + i * (
            K // CORE_NUM // LOOP_SIZE), 0], slicesize=[K // CORE_NUM // LOOP_SIZE, M])
        ub_y = slice_to_ub(gm_y, [block_idx * (K // CORE_NUM) + i * (
            K // CORE_NUM // LOOP_SIZE), 0], slicesize=[K // CORE_NUM // LOOP_SIZE, N])
        ub_x_half = vconv(ub_x, "FP16")
        ub_y_half = vconv(ub_y, "FP16")
        ub_x_nz = nd_to_nz(ub_x_half)
        ub_y_nz = nd_to_nz(ub_y_half)
        l1_x = move_to_l1(ub_x_nz)
        l1_y = move_to_l1(ub_y_nz)
        l0a = move_to_l0A(l1_x, Transpose=True)
        l0b = move_to_l0B(l1_y)
        if i == 0:
            l0c = mmad(l0a, l0b)
        else:
            l0c = mmad(l0a, l0b, l0c)
    ub_out_nz = move_to_ub(l0c)
    ub_out = nz_to_nd(ub_out_nz)
    ub_out = slice_to_ub(ub_out, [0, 0], slicesize=[M, N])
    ub_out = change_view(ub_out, [1, M, N])
    insert_to_gm(gm_out, ub_out, [block_idx, 0, 0], slicesize=[1, M, N])


@sub_kernel(core_num=1)
def add_mm(gm_tmp, gm_out):
    for i in range(CORE_NUM):
        ub_x = slice_to_ub(gm_tmp, [i, 0, 0], slicesize=[1, M, N])
        if i == 0:
            ub_out = ub_x
        else:
            ub_out = vadd(ub_out, ub_x)
    ub_out = reshape(ub_out, [M, N])
    gm_out.load(ub_out)


if __name__ == "__main__":
    gen_golden_data()
    set_context("310P")
    gm_x = Tensor("GM", "FP32", [K, M], format="ND", multi_core=False)
    gm_y = Tensor("GM", "FP32", [K, N], format="ND", multi_core=False)
    gm_out = Tensor("GM", "FP32", [M, N], format="ND", multi_core=False)
    gm_tmp = Tensor("GM", "FP32", [CORE_NUM, M, N],
                    format="ND", multi_core=False)
    compile_func(matmul_294912_2, globals())(gm_x, gm_y, gm_tmp)
    compile_func(add_mm, globals())(gm_tmp, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME, hard_sync=True)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['gm_x', 'gm_y'], outputs=['gm_out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin float32')
    sys.exit(return_code >> 8)
