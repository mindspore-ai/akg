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

import os
if not os.getenv("ENABLE_SWFT_JIT", 0):
    exit()
import mindspore as ms
import numpy as np
import sys
import swft
from pathlib import Path
from swft.core import *
from swft.api import *


TOKENS_LEN = 10001
NUM_HEADS = 1
HEAD_SIZE = 576
NUM_BLOCKS = 16384
BLOCK_SIZE = 16
CORE_NUM = 8


parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from verify_result import verify_result


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
    kvcache_out += 1
    token_len = TOKENS_LEN
    return slot_mapping, kv_in, token_len, kvcache_out

# OP Impl
# ===============================================================================


@swft.jit(core_num=CORE_NUM, nz_args=[1])
def ms_reshape_and_cache_nz(kv_in, kvcache_out, slot_mapping, token_len):
    block_idx = get_block_idx()
    percore_size = ((token_len + CORE_NUM - 1) // CORE_NUM)
    token_num = Scalar("INT32", 0)
    if (block_idx + 1) * percore_size < token_len:
        token_num = percore_size.copy()
    elif block_idx * percore_size < token_len:
        token_num = (token_len - block_idx * percore_size).copy()
    else:
        token_num = Scalar("INT32", 0).copy()
    maxpercore_size = 128
    tile_num = token_num // maxpercore_size
    res_num = token_num % maxpercore_size
    for i in dynamic_loop(tile_num):
        ub_slot = slice_to_ub(slot_mapping, [block_idx * percore_size + i * maxpercore_size], [maxpercore_size])
        for j in dynamic_loop(maxpercore_size):
            offset = move_to_scalar(ub_slot[j])
            x = offset // BLOCK_SIZE
            y = offset % BLOCK_SIZE
            k_ub = slice_to_ub(kv_in, [block_idx * percore_size + i * maxpercore_size + j, 0, 0], [1, NUM_HEADS, HEAD_SIZE])
            k_ub = change_view(k_ub, new_shape=[1, 1, NUM_HEADS * HEAD_SIZE], new_format = "NZ")
            insert_to_gm(kvcache_out, k_ub, [x, y, 0], [1, 1, NUM_HEADS * HEAD_SIZE])
    if res_num > 0:
        ub_slot = slice_to_ub(slot_mapping, [block_idx * percore_size + token_num - res_num], [maxpercore_size])
        for j in dynamic_loop(res_num):
            # 从gm将slot_mapping第n个token的slot值读入ub
            offset = move_to_scalar(ub_slot[j])
            x = offset // BLOCK_SIZE
            y = offset % BLOCK_SIZE
            # 将kv_in的第i个token读入ub
            k_ub = slice_to_ub(
                kv_in, [block_idx * percore_size + token_num - res_num + j, 0, 0], [1, NUM_HEADS, HEAD_SIZE])
            k_ub = change_view(
                k_ub, new_shape=[1, 1, NUM_HEADS * HEAD_SIZE], new_format="NZ")
            insert_to_gm(kvcache_out, k_ub, [x, y, 0], [
                1, 1, NUM_HEADS * HEAD_SIZE])


class Net(ms.nn.Cell):
    def __init__(self) -> None:
        super().__init__()

    def construct(self, kv_in, kvcache_out, slot_mapping, token_len):
        ms_reshape_and_cache_nz(kv_in, kvcache_out, slot_mapping, token_len)
        return kvcache_out + 1


if __name__ == '__main__':
    set_context("310P")
    ms.set_context(mode=ms.GRAPH_MODE)
    np_slot_mapping, np_kv_in, np_token_len, np_kvcache_out = gen_data()
    slot_mapping = ms.Tensor(np_slot_mapping)
    kv_in = ms.Tensor(np_kv_in)
    kvcache_out = ms.Tensor(np.zeros(np_kvcache_out.shape).astype(np_kvcache_out.dtype))
    token_len = np_token_len
    Net = compile_ms_cell(Net)
    net = Net()
    out = net(kv_in, kvcache_out, slot_mapping, token_len)
    sys.exit(verify_result(output=out.numpy(), golden=np_kvcache_out) >> 8)
