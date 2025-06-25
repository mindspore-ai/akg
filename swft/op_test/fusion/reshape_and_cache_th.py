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

OP_NAME = "reshape_and_cache_nz"

os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

MAX_TOKENS = 1024
TOKENS_LEN = 5
NUM_HEADS = 1
HEAD_SIZE = 576
NUM_BLOCKS = 512
BLOCK_SIZE = 16
CORE_NUM = 8

# Numpy Test
# ===============================================================================


def gen_data():
    slot_mapping = np.random.choice(range(
        0, NUM_BLOCKS * BLOCK_SIZE, 1), [TOKENS_LEN], replace=False).astype(np.int32)
    kvcache_out = np.zeros(
        [NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE], dtype=np.float16)
    kv_in = np.random.uniform(-1, 1, [TOKENS_LEN,
                              NUM_HEADS, HEAD_SIZE]).astype(np.float16)
    for i in range(TOKENS_LEN):
        x = slot_mapping[i] // BLOCK_SIZE
        y = slot_mapping[i] % BLOCK_SIZE
        kvcache_out[x][y] = kv_in[i].reshape([NUM_HEADS, HEAD_SIZE])
    kvcache_out = kvcache_out.reshape(
        [NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS * HEAD_SIZE // 16, 16]).transpose(0, 2, 1, 3)
    slot_mapping.tofile(f"./temp/{OP_NAME}/input/slot_mapping.bin")
    kv_in.tofile(f"./temp/{OP_NAME}/input/kv_in.bin")
    token_len = np.array([TOKENS_LEN], dtype=np.int32)
    token_len.tofile(f"./temp/{OP_NAME}/input/token_len.bin")
    kvcache_out.tofile(f"./temp/{OP_NAME}/output/kvcache_out_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=CORE_NUM)
def reshape_and_cache_nz(kv_in, kvcache_out, slot_mapping, token_len):
    block_idx = get_block_idx()
    maxpercore_size = MAX_TOKENS // CORE_NUM
    percore_size = ((token_len + CORE_NUM - 1) // CORE_NUM).copy()
    token_num = Scalar("INT32", 0)
    if (block_idx + 1) * percore_size < token_len:
        token_num.load(percore_size)
    elif block_idx * percore_size < token_len:
        token_num.load(token_len - block_idx * percore_size)
    else:
        token_num.load(Scalar("INT32", 0))
    ub_slot = slice_to_ub(
        slot_mapping, [block_idx * percore_size], [maxpercore_size])
    for i in dynamic_loop(token_num):
        # 从gm将slot_mapping第n个token的slot值读入ub
        offset = move_to_scalar(ub_slot[i])
        x = offset // BLOCK_SIZE
        y = offset % BLOCK_SIZE
        # 将kv_in的第i个token读入ub
        k_ub = slice_to_ub(
            kv_in, [block_idx * percore_size + i, 0, 0], [1, NUM_HEADS, HEAD_SIZE])
        k_ub = change_view(
            k_ub, new_shape=[1, 1, NUM_HEADS * HEAD_SIZE], new_format="NZ")
        insert_to_gm(kvcache_out, k_ub, [x, y, 0], [
            1, 1, NUM_HEADS * HEAD_SIZE])


if __name__ == '__main__':
    set_context("310P")
    gen_data()
    slot_mapping = Tensor(
        "GM", "INT32", [TOKENS_LEN], format="ND", multi_core=False)
    kv_in = Tensor("GM", "FP16", [TOKENS_LEN, NUM_HEADS,
                                  HEAD_SIZE], format="ND", multi_core=False)
    kvcache_out = Tensor("GM", "FP16", [
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS * HEAD_SIZE], format="NZ", multi_core=False)
    token_len = Scalar("INT32")
    compile_func(reshape_and_cache_nz, globals())(
        kv_in, kvcache_out, slot_mapping, token_len)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'kv_in', 'token_len', 'slot_mapping'], outputs=['kvcache_out'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/kvcache_out_actual.bin ./temp/{OP_NAME}/output/kvcache_out_golden.bin float16 4e-2 1e-2 4e-3')
    sys.exit(return_code >> 8)
