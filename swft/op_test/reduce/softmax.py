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

OP_NAME = 'softmax'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

MAX_TOKENS = 16
K = 16384
Scale_Fac = 0.00787402
CORE_NUM = 8
TOKEN_LEN = MAX_TOKENS
# Numpy Test
# ===============================================================================


def gen_golden_data():
    """
    实现 scaleOut = row_max(abs(x)) / 127 和 yOut = round(x / scaleOut)
    :param x: 输入的 numpy 数组
    :return: yOut 计算结果
    """
    # 计算每行元素绝对值的最大值\
    x = np.random.uniform(-10, 10, [TOKEN_LEN, K]).astype(np.float16)
    x_max = np.max(x, -1, keepdims=True)
    x_exp = np.exp(x - x_max).astype(np.float32)
    sum = np.sum(x_exp, -1, keepdims=True)
    y = (x_exp / sum).astype(np.float16)
    # 计算 scaleOut
    x.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    y.tofile(f"./temp/{OP_NAME}/output/gm_y_golden.bin")
# OP Impl


@sub_kernel(core_num=8)
def softmax_impl_npu(gm_input, gm_output):
    # Hardcoded parameters from tiling
    BATCH_SIZE = 16
    DIM = 16384
    BLOCK_DIM = 8
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM  # 2

    # Get swft.core index
    core_idx = get_block_idx()
    start_batch = core_idx * SAMPLES_PER_CORE

    # Process each sample in pipeline
    for i in range(SAMPLES_PER_CORE):
        current_batch = start_batch + i

        # Load input data to UB
        ub_input = slice_to_ub(gm_input, [current_batch, 0], [1, DIM])

        # 1. Find max value (along dim axis)
        ub_max = vcmax(ub_input, reduce_axis=-1)

        # 2. Subtract max value
        ub_sub = vsubs(ub_input, move_to_scalar(ub_max))

        # 3. Compute exp
        ub_exp = vexp(ub_sub)

        # 4. Sum exp values
        ub_sum = vcadd(ub_exp, reduce_axis=-1)

        # 5. Divide each element by sum
        ub_div = vdivs(ub_exp, move_to_scalar(ub_sum))

        # Write result back to GM
        insert_to_gm(gm_output, ub_div, [current_batch, 0], [1, DIM])


if __name__ == "__main__":
    set_context("310P")
    gen_golden_data()
    gm_x = Tensor("GM", "FP16", [MAX_TOKENS, K], format="ND", multi_core=False)
    gm_y = Tensor("GM", "FP16", [MAX_TOKENS, K], format="ND", multi_core=False)
    compile_func(softmax_impl_npu, globals())(gm_x, gm_y)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME, hard_sync=True)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['gm_x'], outputs=['gm_y'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_y_actual.bin ./temp/{OP_NAME}/output/gm_y_golden.bin float16 4e-1 1e-1 4e-1')
    sys.exit(return_code >> 8)
