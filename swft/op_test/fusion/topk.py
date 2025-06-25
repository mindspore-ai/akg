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

OP_NAME = 'topk'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

topk_len = 8
inf = 65500
length = 32

# Numpy Test
# ===============================================================================


def gen_data():
    x = np.random.uniform(-1, 1, [length]).astype(np.float16)
    indices = np.arange(length).astype(np.int32)
    x.tofile(f"./temp/{OP_NAME}/input/x.bin")
    indices.tofile(f"./temp/{OP_NAME}/input/indices.bin")
    for i in range(topk_len):
        res = -inf
        index = -1
        for j in range(i, length):
            if x[j] > res:
                res = x[j]
                index = j
        tmp = x[i]
        x[i] = res
        x[index] = tmp
        tmp = indices[i]
        indices[i] = index
        indices[index] = tmp
    x.tofile(f"./temp/{OP_NAME}/output/x_out_golden.bin")
    indices[:topk_len].astype(np.int32).tofile(
        f"./temp/{OP_NAME}/output/indices_out_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=1)
def topk(x, indices, x_out, indices_out):
    ub_x = move_to_ub(x)
    ub_indices = move_to_ub(indices)
    inf_s = Scalar(ub_x.dtype, -inf)
    neg_one = Scalar("INT32", -1)
    for i in range(topk_len):
        res = inf_s.copy()
        index = neg_one.copy()
        for j in range(i, length):
            x = move_to_scalar(ub_x[j])
            if x > res:
                res.load(x)
                index.load(Scalar("INT32", j))
        tmp = move_to_scalar(ub_x[i])
        ub_x = move_scalar_to_ub(res, ub_x, i)
        ub_x = move_scalar_to_ub(tmp, ub_x, index)
        tmp_i = move_to_scalar(ub_indices[i])
        ub_indices = move_scalar_to_ub(index, ub_indices, i)
        ub_indices = move_scalar_to_ub(tmp_i, ub_indices, index)
    x_out.load(ub_x)
    ub_indices_s = slice_to_ub(ub_indices, [0], [topk_len])
    indices_out.load(ub_indices_s)


if __name__ == "__main__":
    set_context("310P")
    gen_data()
    x = Tensor("GM", "FP16", [length], "ND", False)
    x_out = Tensor("GM", "FP16", [length], "ND", False)
    indices = Tensor("GM", "INT32", [length], "ND", False)
    indices_out = Tensor("GM", "INT32", [topk_len], "ND", False)
    compile_func(topk, globals())(x, indices, x_out, indices_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'x', 'indices'], outputs=['x_out', 'indices_out'], device_id=1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code_1 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/x_out_actual.bin ./temp/{OP_NAME}/output/x_out_golden.bin float16 4e-2 1e-2 4e-3')
    return_code_2 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/indices_out_actual.bin ./temp/{OP_NAME}/output/indices_out_golden.bin int32 4e-1 1e-1 4e-1')
    sys.exit(return_code_1 >> 8 or return_code_2 >> 8)
