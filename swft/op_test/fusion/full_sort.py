import numpy as np
from copy import deepcopy
import os
import sys
from swft.core import *
from swft.api import *

OP_NAME = 'full_sort'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

# Numpy Test

length = 9999
inf = 65500
CORE_NUM = 1


def gen_data():
    x = np.random.randint(0, 256, [length]).astype(np.int32)
    x.tofile(f"./temp/{OP_NAME}/input/x.bin")
    indices = np.argsort(x * (-1), kind="stable")
    output_x = np.sort(x)[::-1]
    output_x.astype(np.float32).tofile(
        f"./temp/{OP_NAME}/output/x_out_golden.bin")
    indices.astype(np.float32).tofile(
        f"./temp/{OP_NAME}/output/indices_out_golden.bin")
    len = np.array([length], dtype=np.int32)
    len.tofile(f"./temp/{OP_NAME}/input/len.bin")


#SWFT

@sub_kernel(core_num=CORE_NUM)
def full_sort_16(x, x_out, indices_out, len):
    idx = get_block_idx()
    if len <= 16:
        if idx == 0:
            ub_x = slice_to_ub(x, [0], [16])
            ub_indices = arange(0, 16, dtype="INT16")
            ub_indices = vconv(ub_indices, "FP16")
            ub_x = vconv(ub_x, "FP16")
            ub_neg = vector_dup(Scalar("FP16", -1), [16], False)
            cond = vcmpvs(ub_indices, len.astype("FP16"), "LT")
            ub_x_mask = where(ub_x, ub_neg, cond)
            concat_local = vconcat(ub_x_mask, ub_indices)
            sorted_local = vsort16(concat_local)
            ub_sorted, ub_index = vextract(sorted_local)
            ub_out = vconv(ub_sorted, "INT32", "a")
            ub_indices_out = vconv(ub_index, "INT32", "a")
            insert_to_gm(x_out, ub_out, [0], [16])
            insert_to_gm(indices_out, ub_indices_out, [0], [16])


@sub_kernel(core_num=CORE_NUM)
def full_sort_64(x, x_out, indices_out, len):
    idx = get_block_idx()
    if len <= 64:
        if idx == 0:
            ub_x = slice_to_ub(x, [0], [64])
            ub_indices = arange(0, 64, dtype="INT16")
            ub_indices = vconv(ub_indices, "FP16")
            ub_x = vconv(ub_x, "FP16")
            ub_neg = vector_dup(Scalar("FP16", -1), [64], False)
            cond = vcmpvs(ub_indices, len.astype("FP16"), "LT")
            ub_x_mask = where(ub_x, ub_neg, cond)
            concat_local = vconcat(ub_x_mask, ub_indices)
            sorted_local = vsort16(concat_local)
            sorted_0 = slice_to_ub(sorted_local, [0], [128])
            sorted_1 = slice_to_ub(sorted_local, [128], [128])
            sorted_2 = slice_to_ub(sorted_local, [256], [128])
            sorted_3 = slice_to_ub(sorted_local, [384], [128])
            sort_tmp_local = vmrgsort4(
                sorted_0, sorted_1, sorted_2, sorted_3, 16, 16, 16, 16)
            ub_sorted, ub_index = vextract(sort_tmp_local)
            ub_out = vconv(ub_sorted, "INT32", "a")
            ub_indices_out = vconv(ub_index, "INT32", "a")
            insert_to_gm(x_out, ub_out, [0], [64])
            insert_to_gm(indices_out, ub_indices_out, [0], [64])


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
def full_sort_init(x, x_out, indice_out, len):
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
        ub_x = slice_to_ub(x, [idx * len_per_round + rep * tiling], [tiling])
        ub_indices = arange(0, tiling, dtype="INT32")
        ub_indices = vadds(
            ub_indices, (idx * len_per_round + rep * tiling).astype("INT32"))
        ub_indices = vconv(ub_indices, "FP32")
        ub_x = vconv(ub_x, "FP32")
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
        ub_neg = vector_dup(Scalar("FP32", -1), [tiling], False)
        cond = vcmpvs(ub_indices, len.astype("FP32"), "LT")
        ub_x_mask = where(ub_x, ub_neg, cond)
        ub_sorted, ub_index = part_sort_256(ub_x_mask, ub_indices)
        insert_to_gm(x_out, ub_sorted, [
                     idx * len_per_round + len_t - res_tiling], [tiling])
        insert_to_gm(indice_out, ub_index, [
                     idx * len_per_round + len_t - res_tiling], [tiling])


@sub_kernel(core_num=1)
def full_sort_mgr(x, indices, x_out, indice_out, len):
    tiling = Scalar("INT32", 256)
    times = Scalar("INT32", 0)
    in_x = Tensor("GM", "FP32", [length], "ND", False)
    in_x.load(x)
    out_x = Tensor("GM", "FP32", [length], "ND", False)
    out_x.load(x_out)
    in_index = Tensor("GM", "FP32", [length], "ND", False)
    in_index.load(indices)
    out_index = Tensor("GM", "FP32", [length], "ND", False)
    out_index.load(indice_out)
    tmp_x = Tensor("GM", "FP32", [length], "ND", False)
    tmp_x.load(in_x)
    tmp_index = Tensor("GM", "FP32", [length], "ND", False)
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


def sort_main(length, x, x_out, indices_out, len, x_tmp=None, indices_tmp=None, sync_0=None):
    if length <= 16:
        compile_func(full_sort_16, globals())(x, x_out, indices_out, len)
        compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    elif length <= 64:
        compile_func(full_sort_64, globals())(x, x_out, indices_out, len)
        compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    else:
        compile_func(full_sort_init, globals())(x, x_out, indices_out, len)
        compile_func(full_sort_mgr, globals())(
            x_out, indices_out, x_tmp, indices_tmp, len)
        compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)


if __name__ == "__main__":
    set_context("310P")
    gen_data()
    x = Tensor("GM", "INT32", [length], "ND", False)
    x_out = Tensor("GM", "FP32", [length], "ND", False)
    indices_out = Tensor("GM", "FP32", [length], "ND", False)
    len = Scalar("INT32")
    x_tmp = Tensor("GM", "FP32", [length], "ND", False)
    indices_tmp = Tensor("GM", "FP32", [length], "ND", False)
    sync_0 = Tensor("GM", "INT32", [64], "ND", False)
    sort_main(length, x, x_out, indices_out, len, x_tmp, indices_tmp)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'x', 'len'], outputs=['x_out', 'indices_out'], device_id=2, profiling=100)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code_1 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/x_out_actual.bin ./temp/{OP_NAME}/output/x_out_golden.bin float32 4e-2 1e-2 4e-3')
    return_code_2 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/indices_out_actual.bin ./temp/{OP_NAME}/output/indices_out_golden.bin float32 1e-4 1e-4 1e-4')
    sys.exit(return_code_1 >> 8 or return_code_2 >> 8)
