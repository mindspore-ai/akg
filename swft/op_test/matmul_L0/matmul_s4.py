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


def gen_data():
    x = np.random.randint(-7, 7, [BS, M, K]).astype(np.int8)
    x_nz = x.reshape(BS, M, K//64, 64).transpose(0, 2, 1, 3)
    y = np.random.randint(-7, 7, [BS, N, K]).astype(np.int8)
    y_nz = y.reshape(BS, N, K//64, 64).transpose(0, 2, 1, 3)
    golden = np.matmul(x.astype(np.int32), y.astype(
        np.int32).transpose(0, 2, 1)).astype(np.int32)
    golden.tofile(f"./temp/{OP_NAME}/output/output_golden.bin")
    x_in = i8toi4(x_nz)
    x_in.tofile(f"./temp/{OP_NAME}/input/input0.bin")
    y_in = i8toi4(y_nz)
    y_in.tofile(f"./temp/{OP_NAME}/input/input1.bin")

# SWFT DSL
# ============================================================


@sub_kernel(core_num=8)
def matmul_s4(input0, input1, output):
    idx = get_block_idx()
    l1_tiling = 3584
    l0_tiling = 256
    for i in dynamic_loop(N // l1_tiling):
        l1_a = slice_to_l1(input0, [idx, 0, 0], [1, M, K])
        l1_a = change_view(l1_a, [M, K])
        l1_b = slice_to_l1(input1, [idx, i * l1_tiling, 0], [1, l1_tiling, K])
        l1_b = change_view(l1_b, [l1_tiling, K])
        l0a = move_to_l0A(l1_a)
        for j in dynamic_loop(l1_tiling // l0_tiling):
            l0b = slice_to_l0B(l1_b, [j * l0_tiling, 0],
                               [l0_tiling, K], transpose=True)
            l0c = mmad(l0a, l0b)
            ub_out = move_to_ub(l0c)
            ub_out = nz_to_nd(ub_out)
            ub_out = change_view(ub_out, [1, M, l0_tiling])
            insert_to_gm(output, ub_out, [
                         idx, 0, i * l1_tiling + j * l0_tiling], [1, M, l0_tiling])


if __name__ == "__main__":
    set_context("310P")
    input0 = Tensor("GM", "INT4", [BS, M, K], "NZ", False)
    input1 = Tensor("GM", "INT4", [BS, N, K], "NZ", False)
    output = Tensor("GM", "INT32", [BS, M, N], "ND", False)
    compile_func(matmul_s4, globals())(input0, input1, output)
    gen_data()
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'input0', 'input1'], outputs=['output'], profiling=100)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/output_actual.bin ./temp/{OP_NAME}/output/output_golden.bin int32')
    sys.exit(return_code >> 8)
