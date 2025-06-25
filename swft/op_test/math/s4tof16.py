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

from swft.core import *
from swft.api import *
import numpy as np
import random
import os
import sys

M = 8
K = 64
N = 8*8192 // 64
EXP = 1
G = 64
OP_NAME = 's4ToFp16'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")


def quant(y_fp16):
    y_fp = y_fp16.reshape((EXP, K // G, G, N))
    y_max = np.max(np.abs(y_fp), keepdims=True, axis=-2)
    scale = y_max / 7.0
    y_int8 = (y_fp / scale).astype(np.int8).reshape((EXP, K, N))
    return y_int8, scale

def i8toi4(y_int8):
    input_x = ((y_int8 + 16) % 16).astype(np.uint8).reshape(-1)
    input_y = (input_x[::2] << 4) | input_x[1::2]
    return input_y

def i4toi8(y_int8):
    input_x_0 = ((y_int8 & 0xf0) >> 4).astype(np.uint16)
    input_x_1 = ((y_int8 & 0x0f)).astype(np.uint16)
    input_x = np.stack([input_x_0, input_x_1]).transpose(1, 0)
    input_x = input_x.reshape((-1, 4, 2048)).transpose(0, 2, 1).reshape(-1)
    input_y_1 = ((input_x[::4] << 4) | (input_x[1::4])).astype(np.uint8)
    input_y_2 = ((input_x[2::4] << 4) | (input_x[3::4])).astype(np.uint8)
    input_y = np.stack([input_y_2, input_y_1]).transpose(1, 0).reshape(-1)
    return input_y

def gen_data():
    y_fp16 = np.random.uniform(-0.3, 0.3, [8*8192]).astype(np.float16)
    y_int8, anti_scale = quant(y_fp16)
    y_nz_int4 = i8toi4(y_int8)
    gm_x = i4toi8(y_nz_int4)
    gm_out = y_int8.astype(np.float16)
    gm_x.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    gm_out.tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")


@sub_kernel(core_num=8)
def conv_s4(gm_x, gm_out):
    block_idx = get_block_idx()
    ub_x = slice_to_ub(gm_x, [block_idx * 2048], slicesize=[2048])
    ub_out = vconv_s42f16(ub_x)
    insert_to_gm(gm_out, ub_out, [block_idx * 8192], [8192])


if __name__ == '__main__':
    set_context("310P")
    gen_data()
    gm_x = Tensor("GM", "INT16", [8*2048], format="ND", multi_core=False)
    gm_out = Tensor("GM", "FP16", [8*8192], format="ND", multi_core=False)
    conv_s4(gm_x, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['gm_x'], outputs=['gm_out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin')
    sys.exit(return_code >> 8)