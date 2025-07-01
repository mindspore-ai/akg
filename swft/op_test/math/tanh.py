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

CORE_NUM = 8
OP_NAME = 'tanh_kernel'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

# Numpy Test
# ===============================================================================


def gen_golden_data():
    x = np.random.uniform(-1, 1, [512]).astype(np.float16)
    out = np.tanh(x)
    x.tofile(f"./temp/{OP_NAME}/input/x.bin")
    out.tofile(f"./temp/{OP_NAME}/output/out_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=CORE_NUM)
def tanh_kernel(x, out):
    x_ub = move_to_ub(x)
    tanh_ub = tanh(x_ub)
    out.load(tanh_ub)


if __name__ == '__main__':
    gen_golden_data()
    set_context("310P")
    x = Tensor("GM", "FP16", [512], format="ND", multi_core=True)
    out = Tensor("GM", "FP16", [512], format="ND", multi_core=True)
    tanh_kernel(x, out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['x'], outputs=['out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/out_actual.bin ./temp/{OP_NAME}/output/out_golden.bin')
    sys.exit(return_code >> 8)
