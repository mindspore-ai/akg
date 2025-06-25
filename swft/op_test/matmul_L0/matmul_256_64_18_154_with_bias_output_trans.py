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

OP_NAME = 'matmul256x64x18x154_withBias_withOutputTrans'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

CORE_NUM = 4096
BATCH = 256
BS = 64
M = 18
K = 154
N = 128

# %998 = "onnx.MatMul"(%arg1, %40) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_29_output:967-9552_7197_1_train_fn_59043:equiv_42_input_ms:147_1457"} : (tensor<256x64x18x154xf32>, tensor<154x128xf32>) -> tensor<256x64x18x128xf32>
# %999 = "onnx.Reshape"(%998, %6) {allowzero = 0 : si64, onnx_node_name = "9552_7197_1_train_fn_59043:equiv_29_output:967-9552_7197_1_train_fn_59043:equiv_42_input_ms:147_499"} : (tensor<256x64x18x128xf32>, tensor<2xi64>) -> tensor<294912x128xf32>
# %1000 = "onnx.Add"(%999, %260) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_35_output:968_1433"} : (tensor<294912x128xf32>, tensor<128xf32>) -> tensor<294912x128xf32>
# %1001 = "onnx.Reshape"(%1000, %127) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_33_output:969_1385"} : (tensor<294912x128xf32>, tensor<4xi64>) -> tensor<256x64x18x128xf32>
# %1002 = "onnx.Transpose"(%1001) {onnx_node_name = "9552_7197_1_train_fn_59043:equiv_376_CNode_57771:970_1401", perm = [0, 2, 1, 3]} : (tensor<256x64x18x128xf32>) -> tensor<256x18x64x128xf32>

# Numpy Test
# ===============================================================================


def gen_golden_data():
    gm_x = np.random.uniform(-0.3, 0.3, [BATCH*BS*M, K]).astype(np.float32)
    gm_y = np.random.uniform(-0.3, 0.3, [K, N]).astype(np.float32)
    in_bias = np.random.uniform(-0.3, 0.3, [N]).astype(np.float32)
    matmul_res = np.matmul(gm_x.astype(np.float16),
                           gm_y.astype(np.float16)).astype(np.float32)
    matmul_res_1 = matmul_res.reshape((BATCH*BS*M, N))
    golden_add = matmul_res_1 + in_bias
    golden_add_1 = golden_add.reshape((BATCH, BS, M, N)).transpose(0, 2, 1, 3)
    gm_x.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    gm_y.tofile(f"./temp/{OP_NAME}/input/gm_y.bin")
    in_bias.tofile(f"./temp/{OP_NAME}/input/in_bias.bin")
    golden_add_1.tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=CORE_NUM)
def matmul256x64x18x154_withBias_withOutputTrans(gm_x, gm_y, gm_bias, gm_out):
    idx = get_block_idx()
    idx_x = idx // 16
    idx_y = idx % 16
    ub_x = slice_to_ub(gm_x, [idx_x*64*M + idx_y * 4 * M, 0], [1*4*M, K])
    ub_x = vconv(ub_x, "FP16")
    ub_x_nz = nd_to_nz(ub_x)
    l1_x = move_to_l1(ub_x_nz)
    l0_a = move_to_l0A(l1_x)
    for k in range(2):
        ub_y = slice_to_ub(gm_y, [0, k * N // 2], [K, N // 2])
        ub_y = vconv(ub_y, "FP16")
        ub_y_nz = nd_to_nz(ub_y)
        l1_y = move_to_l1(ub_y_nz)
        l0_b = move_to_l0B(l1_y)
        ub_bias = slice_to_ub(gm_bias, [k * N // 2], [N // 2])
        l0_c = move_to_l0C(ub_bias, [l0_a.shape[0], l0_b.shape[1]], False)
        l0_c = mmad(l0_a, l0_b, l0_c)
        ub_out_nz = move_to_ub(l0_c)
        ub_out_nd = nz_to_nd(ub_out_nz)
        ub_out = change_view(ub_out_nd, [1, 4, M, N // 2])
        ub_x_trans = transpose(ub_out, [0, 2, 1, 3])  # 1 M 8 N//2
        insert_to_gm(gm_out, ub_x_trans, [
                     idx_x, 0, idx_y * 4, k * N // 2], [1, M, 4, N // 2])


if __name__ == "__main__":
    set_context("310P")
    gen_golden_data()
    gm_x = Tensor("GM", "FP32", [BATCH*BS*M, K], "ND", False)
    gm_y = Tensor("GM", "FP32", [K, N], "ND", False)
    in_bias = Tensor("GM", "FP32", [N], format="ND", multi_core=False)
    gm_out = Tensor("GM", "FP32", [BATCH, M, BS, N], "ND", False)
    compile_func(matmul256x64x18x154_withBias_withOutputTrans,
                 globals())(gm_x, gm_y, in_bias, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME, hard_sync=True)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'gm_x', 'gm_y', 'in_bias'], outputs=['gm_out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin float32')
    sys.exit(return_code >> 8)
