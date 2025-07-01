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

OP_NAME = 'matmul_int8'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

# Numpy Test
# ===============================================================================


def gen_data():
    x = np.random.randint(-120, 120, [18, 64, 128]).astype(np.int8)
    y = np.random.randint(-100, 100, [128, 128]).astype(np.int8)
    out = np.matmul(x.astype(np.int32), y.astype(np.int32)).astype(np.int32)
    x.tofile(f"./temp/{OP_NAME}/input/x.bin")
    y.tofile(f"./temp/{OP_NAME}/input/y.bin")
    out.tofile(f"./temp/{OP_NAME}/output/out_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=6)
def matmul_int8(x, y, out):
    ub_x = move_to_ub(x)
    ub_y = move_to_ub(y)
    ub_y = transpose(ub_y, [1, 0])
    ub_xnz = nd_to_nz(ub_x)
    ub_ynz = nd_to_nz(ub_y)
    l1_x = move_to_l1(ub_xnz)
    l1_y = move_to_l1(ub_ynz)
    l0a = move_to_l0A(l1_x)
    l0b = move_to_l0B(l1_y, Transpose=True)
    l0c = mmad(l0a, l0b)
    ub_out = move_to_ub(l0c)
    ub_out = nz_to_nd(ub_out)
    out.load(ub_out)


if __name__ == '__main__':
    set_context("310P")
    gen_data()
    x = Tensor("GM", "INT8", [18, 64, 128], format="ND", multi_core=True)
    y = Tensor("GM", "INT8", [128, 128], format="ND")
    out = Tensor("GM", "INT32", [18, 64, 128], format="ND", multi_core=True)
    matmul_int8(x, y, out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['x', 'y'], outputs=['out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/out_actual.bin ./temp/{OP_NAME}/output/out_golden.bin int32 4e-2 1e-2 4e-3')
    sys.exit(return_code >> 8)
