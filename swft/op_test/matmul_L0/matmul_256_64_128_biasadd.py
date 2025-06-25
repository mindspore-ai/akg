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

OP_NAME = 'matmul_256_64_128_biasadd'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

BATCH = 256
BS = 64
M = 1
K = 128
N = 2

CORE_NUM = 128
LOOP_SIZE = 8

# %580 = "onnx.Unsqueeze"(%579, %107) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_275_CNode_57778:447_1381"} : (tensor<256x64x128xf32>, tensor<1xi64>) -> tensor<256x64x1x128xf32>
# %582 = "onnx.MatMul"(%580, %45) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_98_output:449-9552_7197_1_train_fn_59043:equiv_211_input_ms:448_920"} : (tensor<256x64x1x128xf32>, tensor<128x2xf32>) -> tensor<256x64x1x2xf32>
# %583 = "onnx.Reshape"(%582, %44) {allowzero = 0 : si64, onnx_node_name = "9552_7197_1_train_fn_59043:equiv_98_output:449-9552_7197_1_train_fn_59043:equiv_211_input_ms:448_138"} : (tensor<256x64x1x2xf32>, tensor<2xi64>) -> tensor<16384x2xf32>
# %584 = "onnx.Add"(%583, %186) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_139_output:450_1251"} : (tensor<16384x2xf32>, tensor<2xf32>) -> tensor<16384x2xf32>
# %585 = "onnx.Reshape"(%584, %96) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_127_output:451_1053"} : (tensor<16384x2xf32>, tensor<4xi64>) -> tensor<256x64x1x2xf32>

# Numpy Test
# ===============================================================================


def gen_golden_data():
    gm_x = np.random.uniform(-0.3, 0.3, [BATCH, BS, K]).astype(np.float32)
    gm_x_unsqueeze = gm_x[:, :, None, :]
    gm_y = np.random.uniform(-0.3, 0.3, [K, N]).astype(np.float32)
    in_bias = np.random.uniform(-0.3, 0.3, [N]).astype(np.float32)
    matmul_res = np.matmul(gm_x_unsqueeze.astype(
        np.float16), gm_y.astype(np.float16)).astype(np.float32)
    matmul_res_1 = matmul_res.reshape((BATCH*BS*M, N))
    golden_add = matmul_res_1 + in_bias
    golden_add_1 = golden_add.reshape((BATCH, BS, M, N))
    gm_x.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    gm_y.tofile(f"./temp/{OP_NAME}/input/gm_y.bin")
    in_bias.tofile(f"./temp/{OP_NAME}/input/in_bias.bin")
    golden_add_1.tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=CORE_NUM)
def matmul_256_64_128_biasadd(gm_x, gm_y, gm_bias, gm_out):
    block_idx = get_block_idx()
    ub_y = move_to_ub(gm_y)
    ub_y_f16 = vconv(ub_y, "FP16")
    ub_y_nz = nd_to_nz(ub_y_f16)
    l1_y = move_to_l1(ub_y_nz)
    l0b = move_to_l0B(l1_y)
    ub_bias = move_to_ub(gm_bias)
    for i in range(LOOP_SIZE):
        ub_x = slice_to_ub(gm_x, [block_idx * (BATCH * BS * M // CORE_NUM) + i * (BATCH * BS * M //
                           CORE_NUM // LOOP_SIZE), 0], slicesize=[(BATCH * BS * M // CORE_NUM // LOOP_SIZE), K])
        ub_x_f16 = vconv(ub_x, "FP16")
        ub_x_nz = nd_to_nz(ub_x_f16)
        l1_x = move_to_l1(ub_x_nz)
        l0a = move_to_l0A(l1_x)
        l0c = mmad(l0a, l0b)
        ub_l0c = move_to_ub(l0c, "FP32")
        ub_bias_l0c = vadd(ub_l0c, ub_bias)
        ub_out = nz_to_nd(ub_bias_l0c)
        insert_to_gm(gm_out, ub_out, [block_idx * (BATCH * BS * M // CORE_NUM) + i * (
            BATCH * BS * M // CORE_NUM // LOOP_SIZE), 0], [BATCH * BS * M // CORE_NUM // LOOP_SIZE, N])


if __name__ == "__main__":
    set_context("310P")
    gen_golden_data()
    gm_x = Tensor("GM", "FP32", [BATCH * BS * M, K],
                  format="ND", multi_core=False)
    gm_y = Tensor("GM", "FP32", [K, N], format="ND", multi_core=False)
    in_bias = Tensor("GM", "FP32", [N], format="ND", multi_core=False)
    gm_out = Tensor("GM", "FP32", [BATCH * BS * M, N],
                    format="ND", multi_core=False)
    compile_func(matmul_256_64_128_biasadd, globals())(
        gm_x, gm_y, in_bias, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME, hard_sync=True)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'gm_x', 'gm_y', 'in_bias'], outputs=['gm_out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin float32')
    sys.exit(return_code >> 8)
