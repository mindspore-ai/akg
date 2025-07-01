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

CORE_NUM = 1
arange_size = 533
OP_NAME = 'arange_kernel'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

# Numpy Test
# ===============================================================================


def gen_golden_data():
    out = np.arange(0, arange_size, 1).astype(np.int32)
    out.tofile(f"./temp/{OP_NAME}/output/out_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=CORE_NUM)
def arange_kernel(out):
    out_ub = arange(0, arange_size, 1, "INT32")
    out.load(out_ub)


if __name__ == '__main__':
    gen_golden_data()
    set_context("310P")
    out = Tensor("GM", "INT32", [arange_size], format="ND", multi_core=False)
    arange_kernel(out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[], outputs=['out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/out_actual.bin ./temp/{OP_NAME}/output/out_golden.bin int32')
    sys.exit(return_code >> 8)
