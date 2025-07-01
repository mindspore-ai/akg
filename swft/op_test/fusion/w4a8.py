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

OP_NAME = 'matmul_s4'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")


BS = 8
M = 16
K = 256
N = 7168

# Numpy Test
# =============================================================


def i8toi4(y_int8):
    input_x = ((y_int8 + 16) % 16).astype(np.uint8).reshape(-1)
    input_y = (input_x[1::2] << 4) | input_x[::2]
    return input_y


def calc_i4(x, y, bias):
    high_x = ((x & 0xf0).astype(np.int8) >> 4).astype(np.int8)
    low_x = ((x & 0x0f) - 8).astype(np.int8)
    low_mm = np.matmul(low_x.astype(np.int32), y.astype(
        np.int32).transpose(0, 2, 1)).astype(np.int32)
    high_mm = np.matmul(high_x.astype(np.int32), y.astype(
        np.int32).transpose(0, 2, 1)).astype(np.int32)
    high_mm = high_mm * 16
    low_mm = low_mm + bias
    return high_mm + low_mm


def gen_data():
    x = np.random.randint(-127, 127, [BS, M, K]).astype(np.int8)
    y = np.random.randint(-7, 7, [BS, N, K]).astype(np.int8)
    bias = np.ones([BS, 1, K]).astype(np.int8) * 8
    bias_y = np.matmul(bias.astype(np.int32), y.astype(
        np.int32).transpose(0, 2, 1)).astype(np.int32)
    y_nz = y.reshape(BS, N, K//64, 64).transpose(0, 2, 1, 3)
    golden = np.matmul(x.astype(np.int32), y.astype(
        np.int32).transpose(0, 2, 1)).astype(np.int32)
    golden_2 = calc_i4(x, y, bias_y)
    print((golden == golden_2).all())
    golden.tofile(f"./temp/{OP_NAME}/output/output_golden.bin")
    x.tofile(f"./temp/{OP_NAME}/input/input0.bin")
    y_in = i8toi4(y_nz)
    y_in.tofile(f"./temp/{OP_NAME}/input/input1.bin")
    bias_y.tofile(f"./temp/{OP_NAME}/input/input2.bin")
# SWFT DSL
# ============================================================


@sub_kernel(core_num=8)
def matmul_w4a8(input0, input1, input2, output):
    idx = get_block_idx()
    l1_tiling = 1024
    l0_tiling = 256
    ub_x = slice_to_ub(input0, [idx, 0, 0], [1, M, K])
    ub_high_x = change_view(ub_x, new_dtype="INT16")
    high_mask = vector_dup(Scalar("INT16", 0xf0f0), [16], False)
    ub_high_x = vand(ub_high_x, high_mask)
    ub_high_x = change_view(ub_high_x, new_dtype="INT8")
    ub_x_half = vconv(ub_high_x, "FP16")
    ub_high_x = vmuls(ub_x_half, 0.0625)
    ub_high_x = vconv(ub_high_x, "INT4")
    ub_high_x = nd_to_nz(ub_high_x)
    l1_high_x = move_to_l1(ub_high_x)
    ub_low_x = change_view(ub_x, new_dtype="INT16")
    low_mask = vector_dup(Scalar("INT16", 0x0f0f), [16], False)
    ub_low_x = vand(ub_low_x, low_mask)
    ub_low_x = change_view(ub_low_x, new_dtype="INT8")
    ub_low_x = vconv(ub_low_x, "FP16")
    ub_low_x = vsubs(ub_low_x, 8)
    ub_low_x = vconv(ub_low_x, "INT4")
    ub_low_x = nd_to_nz(ub_low_x)
    l1_low_x = move_to_l1(ub_low_x)
    for i in dynamic_loop(N // l1_tiling):
        l1_b = slice_to_l1(input1, [idx, i * l1_tiling, 0], [1, l1_tiling, K])
        l1_b = change_view(l1_b, [l1_tiling, K])
        for j in dynamic_loop(l1_tiling // l0_tiling):
            l0a = move_to_l0A(l1_high_x)
            l0b = slice_to_l0B(l1_b, [j * l0_tiling, 0],
                               [l0_tiling, K], transpose=True)
            l0c = mmad(l0a, l0b)
            ub_out_high = move_to_ub(l0c)
            l0a = move_to_l0A(l1_low_x)
            l0c = mmad(l0a, l0b)
            ub_out_low = move_to_ub(l0c)
            ub_out_high = vmuls(ub_out_high, 16)
            ub_scale = slice_to_ub(
                input2, [idx, i * l1_tiling + j * l0_tiling], [1, l0_tiling])
            ub_scale = change_view(ub_scale, [l0_tiling])
            ub_out_low = vadd(ub_out_low, ub_scale)
            ub_out = vadd(ub_out_high, ub_out_low)
            ub_out = nz_to_nd(ub_out)
            insert_to_gm(output, ub_out, [
                         idx, 0, i * l1_tiling + j * l0_tiling], [1, M, l0_tiling])


if __name__ == "__main__":
    gen_data()
    set_context("310P")
    input0 = Tensor("GM", "INT8", [BS, M, K], "ND", False)
    input1 = Tensor("GM", "INT4", [BS, N, K], "NZ", False)
    input2 = Tensor("GM", "INT32", [BS, N], "ND", False)
    output = Tensor("GM", "INT32", [BS, M, N], "ND", False)
    out_low = Tensor("GM", "INT32", [BS, M, N], "ND", False)
    out_high = Tensor("GM", "INT32", [BS, M, N], "ND", False)
    low_x = Tensor("GM", "INT4", [BS, M, K], "NZ", False)
    high_x = Tensor("GM", "INT4", [BS, M, K], "NZ", False)
    compile_func(matmul_w4a8, globals())(input0, input1, input2, output)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'input0', 'input1', 'input2'], outputs=['output'], profiling=100)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/output_actual.bin ./temp/{OP_NAME}/output/output_golden.bin int8')
    sys.exit(return_code >> 8)
