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

OP_NAME = 'moe_init_routing'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

num_rows_actual = 100
HIDDEN_SIZE = 7168
K = 8
EXPERT_NUM = 128
active_num = num_rows_actual * K
CORE_NUM = 8
TILING_ROW = 256
TILING_ROW_2 = 8192
# Numpy Test
# ===============================================================================


def gen_data():
    x = np.random.uniform(-1, 1, size=(num_rows_actual,
                          HIDDEN_SIZE)).astype(np.float16)
    expert_idx = np.random.randint(
        0, EXPERT_NUM, size=(num_rows_actual, K)).astype(np.int32)
    sorted_row_idx = np.argsort(expert_idx.reshape(
        (-1,)), axis=-1, kind="stable").astype(np.int32)
    sorted_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1)
    expert_idx_hist, _ = np.histogram(
        sorted_expert_idx, bins=EXPERT_NUM, range=(0, EXPERT_NUM - 1))
    expert_token_idx = np.cumsum(expert_idx_hist).astype(np.int32)
    expanded_row_idx = np.zeros(sorted_row_idx.shape, dtype=np.int32)
    expanded_row_idx[sorted_row_idx] = np.arange(
        sorted_row_idx.shape[-1], dtype=np.int32)
    expanded_x = x[sorted_row_idx[:active_num] // K, :]
    x.tofile(f"./temp/{OP_NAME}/input/x.bin")
    expert_idx.tofile(f"./temp/{OP_NAME}/input/expert_idx.bin")
    expanded_x.tofile(f"./temp/{OP_NAME}/output/expanded_x_expect.bin")
    expert_token_idx.tofile(
        f"./temp/{OP_NAME}/output/expert_token_idx_expect.bin")
    expanded_row_idx.tofile(
        f"./temp/{OP_NAME}/output/expanded_row_idx_expect.bin")
    num_rows_np = np.array([num_rows_actual], dtype=np.int32)
    num_rows_np.tofile(f"./temp/{OP_NAME}/input/num_rows.bin")
    sorted_row_idx.astype(np.float32).tofile(
        f"./temp/{OP_NAME}/output/sorted_row_idx_expect.bin")
    (sorted_expert_idx * (-1)).astype(np.float32).tofile(
        f"./temp/{OP_NAME}/output/sorted_expert_idx_expect.bin")
    expert_idx_hist.astype(np.int32).tofile(
        f"./temp/{OP_NAME}/output/expert_before_expect.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=1)
def moe_init_routing_16(x, expert_idx, expanded_x, expanded_row_idx, expert_token_idx, expert_before, num_rows):
    len = num_rows * K
    ub_x = slice_to_ub(expert_idx, [0], [16])
    ub_indices = arange(0, 16, dtype="INT32")
    ub_indices = vconv(ub_indices, "FP32")
    ub_x = vconv(ub_x, "FP32")
    ub_neg = vector_dup(Scalar("FP32", 300), [16], False)
    cond = vcmpvs(ub_indices, len.astype("FP32"), "LT")
    ub_x_mask = where(ub_x, ub_neg, cond)
    ub_x_mask = vmuls(ub_x_mask, -1)
    concat_local = vconcat(ub_x_mask, ub_indices)
    sorted_local = vsort16(concat_local)
    ub_sorted, ub_index = vextract(sorted_local)
    ub_sorted = vmuls(ub_sorted, -1)
    ub_tmp = vector_dup(Scalar("INT32", 0), [EXPERT_NUM], False)
    ub_tmp_a = vector_dup(Scalar("INT32", 0), [EXPERT_NUM], False)
    for i in dynamic_loop(len):
        scalar_i = move_to_scalar(ub_sorted[i]).astype("INT32")
        tmp_s = move_to_scalar(ub_tmp[scalar_i])
        ub_tmp = move_scalar_to_ub(tmp_s + 1, ub_tmp, scalar_i)
    expert_before.load(ub_tmp)
    total_s = Scalar("INT32", 0)
    for i in dynamic_loop(EXPERT_NUM):
        tmp_s = move_to_scalar(ub_tmp[i])
        total_s = total_s + tmp_s
        ub_tmp_a = move_scalar_to_ub(total_s, ub_tmp_a, i)
    expert_token_idx.load(ub_tmp_a)
    ub_out = vector_dup(Scalar("INT32", 0), [16], False)
    for i in dynamic_loop(len):
        scale_sortedRowIdx_k = move_to_scalar(ub_index[i]).astype("INT32")
        ub_out = move_scalar_to_ub(
            i.astype("INT32"), ub_out, scale_sortedRowIdx_k)
        scale_sortedRowIdx_div = scale_sortedRowIdx_k / K
        ub_expanded_x_slice = slice_to_ub(
            x, [scale_sortedRowIdx_div, 0], [1, HIDDEN_SIZE])
        insert_to_gm(expanded_x, ub_expanded_x_slice, [i, 0], [1, HIDDEN_SIZE])
    insert_to_gm(expanded_row_idx, ub_out, [0], [16])


@sub_kernel(core_num=1)
def moe_init_routing_64(x, expert_idx, expanded_x, expanded_row_idx, expert_token_idx, expert_before, num_rows):
    len = num_rows * K
    ub_x = slice_to_ub(expert_idx, [0], [64])
    ub_indices = arange(0, 64, dtype="INT32")
    ub_indices = vconv(ub_indices, "FP32")
    ub_x = vconv(ub_x, "FP32")
    ub_neg = vector_dup(Scalar("FP32", 300), [64], False)
    cond = vcmpvs(ub_indices, len.astype("FP32"), "LT")
    ub_x_mask = where(ub_x, ub_neg, cond)
    ub_x_mask = vmuls(ub_x_mask, -1)
    concat_local = vconcat(ub_x_mask, ub_indices)
    sorted_local = vsort16(concat_local)
    sorted_0 = slice_to_ub(sorted_local, [0], [128])
    sorted_1 = slice_to_ub(sorted_local, [128], [128])
    sorted_2 = slice_to_ub(sorted_local, [256], [128])
    sorted_3 = slice_to_ub(sorted_local, [384], [128])
    sort_tmp_local = vmrgsort4(
        sorted_0, sorted_1, sorted_2, sorted_3, 16, 16, 16, 16)
    ub_sorted, ub_index = vextract(sort_tmp_local)
    ub_sorted = vmuls(ub_sorted, -1)
    ub_tmp = vector_dup(Scalar("INT32", 0), [EXPERT_NUM], False)
    ub_tmp_a = vector_dup(Scalar("INT32", 0), [EXPERT_NUM], False)
    for i in dynamic_loop(len):
        scalar_i = move_to_scalar(ub_sorted[i]).astype("INT32")
        tmp_s = move_to_scalar(ub_tmp[scalar_i])
        ub_tmp = move_scalar_to_ub(tmp_s + 1, ub_tmp, scalar_i)
    expert_before.load(ub_tmp)
    total_s = Scalar("INT32", 0)
    for i in dynamic_loop(EXPERT_NUM):
        tmp_s = move_to_scalar(ub_tmp[i])
        total_s = total_s + tmp_s
        ub_tmp_a = move_scalar_to_ub(total_s, ub_tmp_a, i)
    expert_token_idx.load(ub_tmp_a)
    ub_out = vector_dup(Scalar("INT32", 0), [64], False)
    for i in dynamic_loop(len):
        scale_sortedRowIdx_k = move_to_scalar(ub_index[i]).astype("INT32")
        ub_out = move_scalar_to_ub(
            i.astype("INT32"), ub_out, scale_sortedRowIdx_k)
        scale_sortedRowIdx_div = scale_sortedRowIdx_k / K
        ub_expanded_x_slice = slice_to_ub(
            x, [scale_sortedRowIdx_div, 0], [1, HIDDEN_SIZE])
        insert_to_gm(expanded_x, ub_expanded_x_slice, [i, 0], [1, HIDDEN_SIZE])
    insert_to_gm(expanded_row_idx, ub_out, [0], [64])


def part_sort_256(ub_x, ub_indices):
    concat_local = vconcat(ub_x, ub_indices)
    sorted_local = vsort16(concat_local)
    sort_tmp_local_lst = []
    for i in range(4):
        sorted_0 = slice_to_ub(sorted_local, [512 * i], [128])
        sorted_1 = slice_to_ub(sorted_local, [512 * i + 128], [128])
        sorted_2 = slice_to_ub(sorted_local, [512 * i + 256], [128])
        sorted_3 = slice_to_ub(sorted_local, [512 * i + 384], [128])
        sort_tmp_local = vmrgsort4(
            sorted_0, sorted_1, sorted_2, sorted_3, 16, 16, 16, 16)
        sort_tmp_local_lst.append(sort_tmp_local)
    sort_tmp_local = vmrgsort4(sort_tmp_local_lst[0], sort_tmp_local_lst[1],
                               sort_tmp_local_lst[2], sort_tmp_local_lst[3], 64, 64, 64, 64, rep=1)
    ub_sorted, ub_index = vextract(sort_tmp_local)
    return ub_sorted, ub_index


@sub_kernel(core_num=CORE_NUM)
def full_sort_init(x, x_out, indice_out, num_rows):
    len = num_rows * K
    idx = get_block_idx()
    tiling = 256
    len_per_core = (len + CORE_NUM - 1) // CORE_NUM
    len_per_round = (len_per_core + tiling - 1) // tiling * tiling
    len_t = Scalar("INT32", 0)
    if (len_per_round * (idx + 1) <= len):
        len_t = len_per_round.copy()
    elif (len_per_round * idx < len):
        len_t = (len - len_per_round * idx).copy()
    else:
        len_t = Scalar("INT32", 0).copy()
    rep_tiling = len_t // tiling
    res_tiling = len_t % tiling
    for rep in dynamic_loop(rep_tiling):
        ub_indices = arange(0, tiling, dtype="INT32")
        ub_x = slice_to_ub(x, [idx * len_per_round + rep * tiling], [tiling])
        ub_indices = vadds(
            ub_indices, (idx * len_per_round + rep * tiling).astype("INT32"))
        ub_indices = vconv(ub_indices, "FP32")
        ub_x = vconv(ub_x, "FP32")
        ub_x = vmuls(ub_x, -1)
        ub_sorted, ub_index = part_sort_256(ub_x, ub_indices)
        insert_to_gm(x_out, ub_sorted, [
                     idx * len_per_round + rep * tiling], [tiling])
        insert_to_gm(indice_out, ub_index, [
                     idx * len_per_round + rep * tiling], [tiling])
    if (res_tiling > 0):
        ub_x = slice_to_ub(
            x, [idx * len_per_round + len_t - res_tiling], [tiling])
        ub_indices = arange(0, tiling, dtype="INT32")
        ub_indices = vadds(ub_indices, (idx * len_per_round +
                           len_t - res_tiling).astype("INT32"))
        ub_indices = vconv(ub_indices, "FP32")
        ub_x = vconv(ub_x, "FP32")
        ub_x = vmuls(ub_x, -1)
        ub_neg = vector_dup(Scalar("FP32", -300), [tiling], False)
        cond = vcmpvs(ub_indices, len.astype("FP32"), "LT")
        ub_x_mask = where(ub_x, ub_neg, cond)
        ub_sorted, ub_index = part_sort_256(ub_x_mask, ub_indices)
        insert_to_gm(x_out, ub_sorted, [
                     idx * len_per_round + len_t - res_tiling], [tiling])
        insert_to_gm(indice_out, ub_index, [
                     idx * len_per_round + len_t - res_tiling], [tiling])


@sub_kernel(core_num=1)
def full_sort_mgr(x, indices, x_out, indice_out, now_rows):
    len = now_rows * K
    tiling = Scalar("INT32", 256)
    times = Scalar("INT32", 0)
    in_x = Tensor("GM", "FP32", [num_rows_actual * K], "ND", False)
    in_x.load(x)
    out_x = Tensor("GM", "FP32", [num_rows_actual * K], "ND", False)
    out_x.load(x_out)
    in_index = Tensor("GM", "FP32", [num_rows_actual * K], "ND", False)
    in_index.load(indices)
    out_index = Tensor("GM", "FP32", [num_rows_actual * K], "ND", False)
    out_index.load(indice_out)
    tmp_x = Tensor("GM", "FP32", [num_rows_actual * K], "ND", False)
    tmp_x.load(in_x)
    tmp_index = Tensor("GM", "FP32", [num_rows_actual * K], "ND", False)
    tmp_index.load(in_index)
    while (tiling < len):
        tiling = tiling * 4
        rep = (len + tiling - 1) / tiling
        for i in dynamic_loop(rep):
            tmp = tiling.copy()
            if (len - i * tiling < tiling):
                tmp.load(len - i * tiling)
            mgr_sort(slice(in_x, [i * tiling], [tiling]), slice(in_index, [i * tiling], [tiling]),
                     slice(out_x, [i * tiling], [tiling]), slice(out_index, [i * tiling], [tiling]), tmp, tiling)

        tmp_x.load(in_x)
        in_x.load(out_x)
        out_x.load(tmp_x)
        tmp_index.load(in_index)
        in_index.load(out_index)
        out_index.load(tmp_index)
        times.load(times + Scalar("INT32", 1))

    if (times % 2 == 1):
        tiling_t = 4096
        rep_tiling = len // tiling_t
        res_tiling = len % tiling_t
        for rep in dynamic_loop(rep_tiling):
            ub_x = slice_to_ub(in_x, [rep * tiling_t], [tiling_t])
            insert_to_gm(x, ub_x, [rep * tiling_t], [tiling_t])
            ub_indices = slice_to_ub(in_index, [rep * tiling_t], [tiling_t])
            insert_to_gm(indices, ub_indices, [rep * tiling_t], [tiling_t])

        if (res_tiling > 0):
            ub_x = slice_to_ub(in_x, [len - res_tiling], [tiling_t])
            insert_to_gm(x, ub_x, [len - res_tiling], [tiling_t])
            ub_indices = slice_to_ub(in_index, [len - res_tiling], [tiling_t])
            insert_to_gm(indices, ub_indices, [len - res_tiling], [tiling_t])


@sub_kernel(core_num=CORE_NUM)
def moe_init_routing(sorted_exp, sorted_row_idx, x, expanded_x, expanded_row_idx, expert_token_idx_tmp, expert_before_tmp, num_rows):
    idx = get_block_idx()
    total_len = num_rows * K
    len_per_core = ((total_len + 7) // CORE_NUM + 7) // 8 * 8
    len = Scalar("INT32", 0)
    if len_per_core * (idx + 1) <= total_len:
        len = len_per_core.copy()
    elif len_per_core * idx < total_len:
        len = (total_len - len_per_core * idx).copy()
    else:
        len = Scalar("INT32", 0).copy()
    rep = (len + TILING_ROW - 1) // TILING_ROW
    ub_sum = vector_dup(Scalar("INT32", 0), [EXPERT_NUM], False)
    ub_before = vector_dup(Scalar("INT32", 0), [EXPERT_NUM], False)
    last_s = Scalar("INT32", -1)
    tmp_s = Scalar("INT32", 0)
    for i in dynamic_loop(rep):
        tiling = Scalar("INT32", TILING_ROW)
        if (len - i * TILING_ROW < TILING_ROW):
            tiling = (len - i * TILING_ROW).copy()
        ub_sorted = slice_to_ub(
            sorted_exp, [idx * len_per_core + i * TILING_ROW], [TILING_ROW])
        ub_sorted = vmuls(ub_sorted, -1)
        for j in dynamic_loop(tiling):
            now_s = move_to_scalar(ub_sorted[j]).astype("INT32")
            if (now_s != last_s):
                if (last_s != Scalar("INT32", -1)):
                    ub_before = move_scalar_to_ub(
                        i * TILING_ROW + j - tmp_s, ub_before, last_s)
                    loop_s = now_s - last_s
                    for k in dynamic_loop(loop_s):
                        ub_sum = move_scalar_to_ub(
                            i * TILING_ROW + j, ub_sum, last_s + k)
                last_s = now_s
                tmp_s = i * TILING_ROW + j
    if (last_s != Scalar("INT32", -1)):
        ub_before = move_scalar_to_ub(len - tmp_s, ub_before, last_s)
        loop_s = EXPERT_NUM - last_s
        for k in dynamic_loop(loop_s):
            ub_sum = move_scalar_to_ub(len, ub_sum, last_s + k)
    insert_to_gm(expert_token_idx_tmp, ub_sum, [
                 idx * EXPERT_NUM], [EXPERT_NUM])
    insert_to_gm(expert_before_tmp, ub_before, [
                 idx * EXPERT_NUM], [EXPERT_NUM])

    ub_sortedRowIdx = slice_to_ub(
        sorted_row_idx, [idx * len_per_core], [TILING_ROW])
    for i in dynamic_loop(len):
        scale_sortedRowIdx_k = move_to_scalar(
            ub_sortedRowIdx[i]).astype("INT32")
        scale_sortedRowIdx_div = scale_sortedRowIdx_k / K
        ub_expanded_x_slice = slice_to_ub(
            x, [scale_sortedRowIdx_div, 0], [1, HIDDEN_SIZE])
        insert_to_gm(expanded_x, ub_expanded_x_slice, [
                     idx * len_per_core + i, 0], [1, HIDDEN_SIZE])

    rep_2 = (len + TILING_ROW_2 - 1) // TILING_ROW_2
    for i in dynamic_loop(rep_2):
        tiling_2 = Scalar("INT32", TILING_ROW_2)
        if (len - i * TILING_ROW_2 < TILING_ROW_2):
            tiling_2 = (len - i * TILING_ROW_2).copy()
        ub_out = vector_dup(Scalar("INT32", 0), [TILING_ROW_2], False)
        rep = (total_len + TILING_ROW - 1) // TILING_ROW
        for j in dynamic_loop(rep):
            tiling = Scalar("INT32", TILING_ROW)
            if (total_len - j * TILING_ROW < TILING_ROW):
                tiling = (total_len - j * TILING_ROW).copy()
            ub_sortedRowIdx = slice_to_ub(
                sorted_row_idx, [j * TILING_ROW], [tiling])
            for k in dynamic_loop(tiling):
                scale_sortedRowIdx_k = move_to_scalar(
                    ub_sortedRowIdx[k]).astype("INT32")
                if scale_sortedRowIdx_k < idx * len_per_core + i * TILING_ROW_2 + tiling_2 and scale_sortedRowIdx_k >= idx * len_per_core + i * TILING_ROW_2:
                    ub_out = move_scalar_to_ub(
                        (j * TILING_ROW + k).astype("INT32"), ub_out, scale_sortedRowIdx_k - idx * len_per_core - i * TILING_ROW_2)
        insert_to_gm(expanded_row_idx, ub_out, [
                     idx * len_per_core + i * TILING_ROW_2], [tiling_2])


@sub_kernel(core_num=8)
def moe_init_add(expert_token_idx_tmp, expert_token_idx, expert_before_tmp, expert_before):
    idx = get_block_idx()
    exp_per_core = EXPERT_NUM // 8
    ub_out = vector_dup(Scalar("INT32", 0), [exp_per_core], False)
    for i in dynamic_loop(CORE_NUM):
        ub_in = slice_to_ub(expert_token_idx_tmp, [
                            idx * exp_per_core + i * EXPERT_NUM], [exp_per_core])
        ub_out = vadd(ub_out, ub_in)
    insert_to_gm(expert_token_idx, ub_out, [
                 idx * exp_per_core], [exp_per_core])
    ub_tmp = vector_dup(Scalar("INT32", 0), [exp_per_core], False)
    for i in dynamic_loop(CORE_NUM):
        ub_in = slice_to_ub(expert_before_tmp, [
                            idx * exp_per_core + i * EXPERT_NUM], [exp_per_core])
        ub_tmp = vadd(ub_tmp, ub_in)
    insert_to_gm(expert_before, ub_tmp, [
                 idx * exp_per_core], [exp_per_core])


@sub_kernel(core_num=CORE_NUM)
def moe_init_routing_expand_x(sorted_row_idx, x, expanded_x, num_rows):
    len = num_rows * K
    idx = get_block_idx()
    len_per_0 = (len + CORE_NUM - 1) // CORE_NUM
    tiling = TILING_ROW
    len_per_core = (len_per_0 + tiling - 1) // tiling * tiling
    now_len = Scalar("INT32", 0)
    if len_per_core * (idx + 1) <= len:
        now_len = len_per_core.copy()
    elif len_per_core * idx < len:
        now_len = (len - len_per_core * idx).copy()
    else:
        now_len = Scalar("INT32", 0).copy()
    rep = now_len // tiling
    res = now_len % tiling
    for i in dynamic_loop(rep):
        ub_sorted_row_idx = slice_to_ub(
            sorted_row_idx, [idx * len_per_core + i * tiling], [tiling])
        for j in dynamic_loop(tiling):
            scalar_idx = move_to_scalar(ub_sorted_row_idx[j]).astype("INT32")
            scalar_idx = scalar_idx // K
            ub_expand_x = slice_to_ub(x, [scalar_idx, 0], [1, HIDDEN_SIZE])
            insert_to_gm(expanded_x, ub_expand_x, [
                         idx * len_per_core + i * tiling + j, 0], [1, HIDDEN_SIZE])
    if (res > 0):
        ub_sorted_row_idx = slice_to_ub(
            sorted_row_idx, [idx * len_per_core + now_len - res], [tiling])
        for j in dynamic_loop(res):
            scalar_idx = move_to_scalar(ub_sorted_row_idx[j]).astype("INT32")
            scalar_idx = scalar_idx // K
            ub_expand_x = slice_to_ub(x, [scalar_idx, 0], [1, HIDDEN_SIZE])
            insert_to_gm(expanded_x, ub_expand_x, [
                         idx * len_per_core + now_len - res + j, 0], [1, HIDDEN_SIZE])


def moe_init_routing_main(x, expert_idx, expanded_x, expanded_row_idx, expert_token_idx, expert_before, num_rows,
                          x_tmp=None, indices_tmp=None, sorted_exp=None, sorted_row_idx=None, expert_token_idx_tmp=None,
                          expert_before_tmp=None):
    if num_rows_actual <= 2:
        compile_func(moe_init_routing_16, globals())(x, expert_idx, expanded_x,
                                                     expanded_row_idx, expert_token_idx, expert_before, num_rows)
        compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    elif num_rows_actual <= 8:
        compile_func(moe_init_routing_64, globals())(x, expert_idx, expanded_x,
                                                     expanded_row_idx, expert_token_idx, expert_before, num_rows)
        compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    else:
        compile_func(full_sort_init, globals())(
            expert_idx, sorted_exp, sorted_row_idx, num_rows)
        compile_func(full_sort_mgr, globals())(
            sorted_exp, sorted_row_idx, x_tmp, indices_tmp, num_rows)
        compile_func(moe_init_routing, globals())(
            sorted_exp, sorted_row_idx, x, expanded_x, expanded_row_idx, expert_token_idx_tmp, expert_before_tmp, num_rows)
        compile_func(moe_init_routing_expand_x, globals())(
            sorted_row_idx, x, expanded_x, num_rows)
        compile_func(moe_init_add, globals())(
            expert_token_idx_tmp, expert_token_idx, expert_before_tmp, expert_before)
        compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce",
                       OP_NAME, hard_sync=True)


if __name__ == '__main__':
    set_context("310P", "ASCENDC")
    gen_data()
    x = Tensor("GM", "FP16", [num_rows_actual, HIDDEN_SIZE],
               format="ND", multi_core=False)
    expert_idx = Tensor(
        "GM", "INT32", [num_rows_actual * K], format="ND", multi_core=False)
    expanded_x = Tensor(
        "GM", "FP16", [active_num, HIDDEN_SIZE], format="ND", multi_core=False)
    expanded_row_idx = Tensor(
        "GM", "INT32", [active_num], format="ND", multi_core=False)
    expert_token_idx = Tensor(
        "GM", "INT32", [EXPERT_NUM], format="ND", multi_core=False)
    expert_before = Tensor(
        "GM", "INT32", [EXPERT_NUM], format="ND", multi_core=False)
    sorted_row_idx = Tensor(
        "GM", "FP32", [num_rows_actual * K], format="ND", multi_core=False)
    sorted_exp = Tensor(
        "GM", "FP32", [num_rows_actual * K], format="ND", multi_core=False)
    x_tmp = Tensor("GM", "FP32", [num_rows_actual * K],
                   format="ND", multi_core=False)
    indices_tmp = Tensor("GM", "FP32", [num_rows_actual * K],
                         format="ND", multi_core=False)
    expert_token_idx_tmp = Tensor(
        "GM", "INT32", [EXPERT_NUM * CORE_NUM], format="ND", multi_core=False)
    expert_before_tmp = Tensor(
        "GM", "INT32", [EXPERT_NUM * CORE_NUM], format="ND", multi_core=False)
    sync_0_tmp = Tensor("GM", "FP32", [64], format="ND", multi_core=False)
    sync_1_tmp = Tensor("GM", "FP32", [64], format="ND", multi_core=False)
    num_rows = Scalar("INT32")
    moe_init_routing_main(x, expert_idx, expanded_x, expanded_row_idx,
                          expert_token_idx, expert_before, num_rows, x_tmp, indices_tmp, sorted_exp, sorted_row_idx, expert_token_idx_tmp, expert_before_tmp)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['x', 'expert_idx', 'num_rows'], device_id=0, outputs=[
                'expert_token_idx', 'expanded_row_idx', 'expanded_x', 'expert_before'], profiling=100)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code_1 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/expert_token_idx_actual.bin ./temp/{OP_NAME}/output/expert_token_idx_expect.bin int32 1e-4 1e-4 4e-3')
    return_code_2 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/expanded_row_idx_actual.bin ./temp/{OP_NAME}/output/expanded_row_idx_expect.bin int32 1e-4 1e-4 4e-4')
    return_code_3 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/expanded_x_actual.bin ./temp/{OP_NAME}/output/expanded_x_expect.bin float16 1e-4 1e-4 4e-4')
    return_code_4 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/expert_before_actual.bin ./temp/{OP_NAME}/output/expert_before_expect.bin int32 1e-4 1e-4 4e-3')
    sys.exit(return_code_1 >> 8 or return_code_2 >>
             8 or 0 >> 8 or return_code_4 >> 8)
