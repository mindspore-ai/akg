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

OP_NAME = 'reduce'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

BLOCK_DIM = 8
reduce_axis = -1
static_cols = 65536
row = 2

# Numpy Test
# ===============================================================================


def gen_golden_data():
    gm_input = np.random.uniform(-1, 1, [row, static_cols]).astype(np.float16)
    gm_output = np.sum(gm_input, -1).astype(np.float16)
    tiling = np.zeros([128], dtype=np.int32)
    tiling[0] = row
    gm_input.tofile(f"./temp/{OP_NAME}/input/gm_input.bin")
    gm_output.tofile(f"./temp/{OP_NAME}/output/gm_output_golden.bin")
    tiling.tofile(f"./temp/{OP_NAME}/input/tiling.bin")


@sub_kernel(core_num=BLOCK_DIM)
def reduce_sum_op_impl_npu(gm_input, gm_output, tiling):
    tiling_row = 16
    tiling_col = 1024

    ub_tiling = move_to_ub(tiling)
    rows = move_to_scalar(ub_tiling[0])

    block_idx = get_block_idx()
    rows_per_core = (rows + BLOCK_DIM - 1) // BLOCK_DIM
    rows_per_core = (rows_per_core + tiling_row - 1) // tiling_row * tiling_row
    row_num = Scalar("INT32", 0)
    if (block_idx + 1) * rows_per_core < rows:
        row_num.load(rows_per_core)
    elif block_idx * rows_per_core < rows:
        row_num.load(rows - block_idx * rows_per_core)
    else:
        row_num.load(Scalar("INT32", 0))
    start_row = block_idx * rows_per_core
    for i in dynamic_loop(row_num // tiling_row):
        ub_out = vector_dup(Scalar("FP16", 0), [tiling_row], False)
        for j in dynamic_loop(tiling_row):
            ub_tmp = vector_dup(Scalar("FP32", 0), [
                                static_cols // tiling_col], False)
            for k in dynamic_loop(static_cols // tiling_col):
                ub_input = slice_to_ub(
                    gm_input, [start_row + i * tiling_row + j, k * tiling_col], [1, tiling_col])
                ub_input_fp32 = vconv(ub_input, "FP32")
                ub_output_fp32 = vcadd(ub_input_fp32, reduce_axis=reduce_axis)
                sum_tmp = move_to_scalar(ub_output_fp32)
                ub_tmp = move_scalar_to_ub(sum_tmp, ub_tmp, k)
            ub_output_fp32 = vcadd(ub_tmp, reduce_axis=reduce_axis)
            ub_output = vconv(ub_output_fp32, "FP16")
            sum = move_to_scalar(ub_output)
            ub_out = move_scalar_to_ub(sum, ub_out, j)
        insert_to_gm(gm_output, ub_out, [
                     start_row + i * tiling_row], [tiling_row])

    res_num = row_num % tiling_row
    if res_num > 0:
        ub_out = vector_dup(Scalar("FP16", 0), [tiling_row], False)
        for j in dynamic_loop(res_num):
            ub_tmp = vector_dup(Scalar("FP32", 0), [
                                static_cols // tiling_col], False)
            for k in dynamic_loop(static_cols // tiling_col):
                ub_input = slice_to_ub(
                    gm_input, [start_row + row_num - res_num + j, k * tiling_col], [1, tiling_col])
                ub_input_fp32 = vconv(ub_input, "FP32")
                ub_output_fp32 = vcadd(ub_input_fp32, reduce_axis=reduce_axis)
                sum_tmp = move_to_scalar(ub_output_fp32)
                ub_tmp = move_scalar_to_ub(sum_tmp, ub_tmp, k)
            ub_output_fp32 = vcadd(ub_tmp, reduce_axis=reduce_axis)
            ub_output = vconv(ub_output_fp32, "FP16")
            sum = move_to_scalar(ub_output)
            ub_out = move_scalar_to_ub(sum, ub_out, j)
        insert_to_gm(gm_output, ub_out, [
                     start_row + row_num - res_num], [tiling_row])


def reduce_sum_op_host_run():
    set_context("310P")

    # Define input and output tensors
    gm_input = Tensor(
        "GM", "FP16", [row, static_cols], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [row], format="ND", multi_core=False)
    tiling = Tensor("GM", "INT32", [128], format="ND", multi_core=False)

    # Execute the NPU kernel
    compile_func(reduce_sum_op_impl_npu, globals())(
        gm_input, gm_output, tiling)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    gen_golden_data()
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'gm_input', 'tiling'], outputs=['gm_output'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_output_actual.bin ./temp/{OP_NAME}/output/gm_output_golden.bin float16 4e-2 1e-2 4e-3')
    return return_code


if __name__ == '__main__':
    return_code = reduce_sum_op_host_run()
    sys.exit(return_code >> 8)
