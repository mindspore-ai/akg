from swft.core import *
from swft.api import *
import numpy as np
import os
import random
import sys

os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "32"

OP_NAME = 'grouped_matmul_3584_1024_transpose'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

EXP = 256
K = 7168
G = 256
N = 1024
M = 8
CORE_NUM = 8

#Numpy Test

def quant(y_int):
    y_fp = y_int.reshape((EXP, K // G, G, N))
    y_max = np.max(np.abs(y_fp), keepdims=True, axis=-2)
    scale = (y_max.astype(np.float32) / 7.0)
    y_int8 = np.round(y_fp.astype(np.float32) /
                      scale).astype(np.int8).reshape((EXP, K, N))
    return y_int8, scale


def i8toi4(y_int8):
    input_x = ((y_int8 + 16) % 16).astype(np.uint8).reshape(-1)
    input_y = (input_x[1::2] << 4) | input_x[::2]
    return input_y


def dyn_quant(x_fp16):
    x_abs = np.abs(x_fp16)
    x_max = np.max(x_abs, axis=-1, keepdims=True)
    anti_scale = x_max.astype(np.float32) / 127.0
    x_int8 = np.round(x_fp16.astype(np.float32) / anti_scale).astype(np.int8)
    return x_int8, anti_scale


def gen_data():
    x_fp16 = np.random.uniform(-0.3, 0.3, [M, K]).astype(np.float16)
    x_int8, x_scale = dyn_quant(x_fp16)
    y_int_O = np.random.uniform(-0.3, 0.3, [EXP, K, N]).astype(np.float16)
    y_int8, y_scale = quant(y_int_O)
    y_nz_int8 = y_int8.transpose(0, 2, 1).reshape(
        EXP, N, K//64, 64).transpose(0, 2, 1, 3)
    y_nz_int4 = i8toi4(y_nz_int8)
    gl = np.zeros([EXP], dtype=np.int32)
    out = np.zeros([M, N], dtype=np.float16)
    y_fp16 = y_int8.reshape((EXP, K // G, G, N)).astype(np.float16) * y_scale
    before = 0
    for i in range(EXP):
        gl[i] = random.randint(before, min(M, before + (M + EXP - 1) // EXP * 2))
        for j in range(gl[i] - before):
            for k in range(K // G):
                out[before + j, :] += np.matmul(x_int8[before + j, k*G:(k+1)*G].astype(np.float16), y_fp16[i, k, :, :].reshape(
                    [G, N]).astype(np.float16)).astype(np.float16)
            out[before + j, :] = out[before + j, :] * x_scale[before + j, :]
        before = gl[i]
    bias = np.ones([EXP, 1, K]).astype(np.float16) * 8
    bias_y = np.matmul(bias, (y_fp16).reshape((EXP, K, N))).astype(np.float32)
    x_int8.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    y_nz_int4.tofile(f"./temp/{OP_NAME}/input/gm_weight.bin")
    gl.tofile(f"./temp/{OP_NAME}/input/group_list.bin")
    y_scale.tofile(f"./temp/{OP_NAME}/input/gm_y_scale.bin")
    x_scale.tofile(f"./temp/{OP_NAME}/input/gm_x_scale.bin")
    bias_y.tofile(f"./temp/{OP_NAME}/input/gm_bias.bin")
    out.astype(np.float16).tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")

#SWFT

@sub_kernel(core_num=CORE_NUM)
def grouped_matmul_3584_1024_transpose(x, w, group_list, x_scale, y_scale, bias, out):
    block_idx = get_block_idx()
    ub_list = slice_to_ub(group_list, [0], slicesize=[EXP])
    ub_tiling_size = 1024
    ub_tiling = K // ub_tiling_size
    start_batch = Scalar("INT32", 0)
    l0b_tiling_size = 256
    l0b_tiling = ub_tiling_size // l0b_tiling_size
    for i in dynamic_loop(EXP):
        end_batch = move_to_scalar(ub_list[i])
        if start_batch == end_batch:
            continue
        w_l1 = slice_to_l1(w, [i, block_idx * N // 8, 0], [1, N // 8, K])
        w_l1 = change_view(w_l1, [N // 8, K])
        for j in dynamic_loop(end_batch - start_batch):
            ub_scale_x = slice_to_ub(x_scale, [start_batch + j], [1])
            ub_out_high_new = vector_dup(Scalar("FP32", 0), [1, N // 8], False)
            ub_out_high_new = change_view(ub_out_high_new, new_format="NZ")
            ub_out_low_new = vector_dup(Scalar("FP32", 0), [1, N // 8], False)
            ub_out_low_new = change_view(ub_out_low_new, new_format="NZ")
            for n in dynamic_loop(ub_tiling):
                ub_x = slice_to_ub(x, [start_batch + j, n * ub_tiling_size], [1, ub_tiling_size])
                ub_high_x = change_view(ub_x, new_shape=[1, ub_tiling_size // 2], new_dtype="INT16")
                high_mask = vector_dup(Scalar("INT16", 0xf0f0), [16], False)
                ub_high_x = vand(ub_high_x, high_mask)
                ub_high_x = change_view(ub_high_x, new_shape=[1, ub_tiling_size], new_dtype="INT8", new_format="NZ")
                ub_x_half = vconv(ub_high_x, "FP16")
                ub_high_x = vmuls(ub_x_half, 0.0625)
                ub_high_x = vconv(ub_high_x, "INT4")
                ub_high_x = pad_to_ub(ub_high_x, [16, ub_tiling_size])
                l1_high_x = move_to_l1(ub_high_x)
                ub_low_x = change_view(ub_x, new_shape=[1, ub_tiling_size // 2], new_dtype="INT16")
                low_mask = vector_dup(Scalar("INT16", 0x0f0f), [16], False)
                ub_low_x = vand(ub_low_x, low_mask)
                ub_low_x = change_view(ub_low_x, new_shape=[1, ub_tiling_size], new_dtype="INT8", new_format="NZ")
                ub_low_x = vconv(ub_low_x, "FP16")
                ub_low_x = vsubs(ub_low_x, 8)
                ub_low_x = vconv(ub_low_x, "INT4")
                ub_low_x = pad_to_ub(ub_low_x, [16, ub_tiling_size])
                l1_low_x = move_to_l1(ub_low_x)
                for k in range(l0b_tiling):
                    x_l0a_high = slice_to_l0A(
                            l1_high_x, [0, k * l0b_tiling_size], [16, l0b_tiling_size])
                    w_l0b = slice_to_l0B(
                        w_l1, [0, n * ub_tiling_size + k * l0b_tiling_size], [N//8, l0b_tiling_size], transpose=True)
                    l0c_high = mmad(x_l0a_high, w_l0b)
                    ub_out_high = move_to_ub(l0c_high)
                    ub_out_high = vconv(ub_out_high, "FP32")
                    x_l0a_low = slice_to_l0A(
                        l1_low_x, [0, k * l0b_tiling_size], [16, l0b_tiling_size])
                    l0c_low = mmad(x_l0a_low, w_l0b)
                    ub_out_low = move_to_ub(l0c_low)
                    ub_out_low = vconv(ub_out_low, "FP32")
                    ub_y_scale = slice_to_ub(
                        y_scale, [i, n*l0b_tiling + k, 0, block_idx * N//8], [1, 1, 1, N // 8])
                    ub_y_scale_cv = change_view(ub_y_scale, [N//8], "NZ")
                    ub_high_slice = slice_to_ub(ub_out_high, [0, 0], [1, N//8])
                    ub_out_high_mul = vmul(ub_high_slice, ub_y_scale_cv)
                    ub_low_slice = slice_to_ub(ub_out_low, [0, 0], [1, N // 8])
                    ub_out_low_mul = vmul(ub_low_slice, ub_y_scale_cv)
                    ub_out_high_new = vadd(ub_out_high_new, ub_out_high_mul)
                    ub_out_low_new = vadd(ub_out_low_new, ub_out_low_mul)
            ub_out_high = vmuls(ub_out_high_new, 16)
            ub_out_new = vadd(ub_out_high, ub_out_low_new)
            ub_out_nd = change_view(ub_out_new, new_format="ND")
            ub_bias = slice_to_ub(bias, [i, block_idx * N // 8], [1, N // 8])
            ub_out_add = vadd(ub_out_nd, ub_bias)
            ub_scale_scalar = move_to_scalar(ub_scale_x)
            ub_final_out = vmuls(ub_out_add, ub_scale_scalar)
            ub_final_out = vconv(ub_final_out, "FP16")
            insert_to_gm(out, ub_final_out, [start_batch + j, block_idx * N // 8], [1, N // 8])
    
        start_batch = end_batch.copy()

if __name__ == "__main__":
    set_context("310P")
    gen_data()
    gm_x = Tensor("GM", "INT8", [M, K], "ND", False)
    gm_weight = Tensor("GM", "INT4", [EXP, N, K], "NZ", False)
    gm_bias = Tensor("GM", "FP32", [EXP, N], "ND", False)
    group_list = Tensor("GM", "INT32", [EXP], "ND", False)
    gm_x_scale = Tensor("GM", "FP32", [M], "ND", False)
    gm_y_scale = Tensor("GM", "FP32", [EXP, K // G, 1, N], "ND", False)
    gm_out = Tensor("GM", "FP16", [M, N], "ND", False)
    compile_func(grouped_matmul_3584_1024_transpose, globals())(gm_x, gm_weight,
                group_list, gm_x_scale, gm_y_scale, gm_bias, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'gm_x', 'gm_weight', 'group_list', 'gm_x_scale', 'gm_y_scale', 'gm_bias'], outputs=['gm_out'], profiling=100)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin float16 4e-2 1e-2 4e-3')
    sys.exit(return_code >> 8)
