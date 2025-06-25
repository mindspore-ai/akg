from swft.core import *
from swft.api import *
import numpy as np
import random
import os
import sys

os.environ["OMP_NUM_THREADS"] = "64"  # OpenMP 线程数
os.environ["MKL_NUM_THREADS"] = "32"  # MKL 线程数

OP_NAME = 'gmm_w4a8_7168_512'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

EXP = 256
K = 7168
G = 64
N = 512
M = 8
CORE_NUM = 8

# Numpy Test
# ======================================================================


def quant(y_int):
    y_fp = y_int.reshape((EXP, K // G, G, N))
    y_max = np.max(np.abs(y_fp), keepdims=True, axis=-2)
    scale = (y_max.astype(np.float32) / 7.0).astype(np.float16)
    y_int8 = np.round(y_fp / scale).astype(np.int8).reshape((EXP, K, N))
    return y_int8, scale


def i8toi4(y_int8):
    input_x = ((y_int8 + 16) % 16).astype(np.uint8).reshape(-1)
    input_y = (input_x[::2] << 4) | input_x[1::2]
    return input_y


def i4toi8(y_int8):
    input_x_0 = ((y_int8 & 0xf0) >> 4).astype(np.uint16)
    input_x_1 = ((y_int8 & 0x0f)).astype(np.uint16)
    input_x = np.stack([input_x_0, input_x_1]).transpose(1, 0)
    input_x = input_x.reshape((-1, 224, 8, 64, 32)).transpose(0,
                                                              2, 1, 3, 4).reshape((-1, 4, 7168)).transpose(0, 2, 1)
    input_x = input_x.reshape(
        (-1, 8, 224, 64, 32)).transpose(0, 2, 1, 3, 4).reshape(-1)
    input_y_1 = ((input_x[::4] << 4) | (input_x[1::4])).astype(np.uint8)
    input_y_2 = ((input_x[2::4] << 4) | (input_x[3::4])).astype(np.uint8)
    input_y = np.stack([input_y_2, input_y_1]).transpose(1, 0).reshape(-1)
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
    y_transpose_nz_int8 = y_int8.transpose(0, 2, 1).reshape(
        EXP, N, K//32, 32).transpose(0, 2, 1, 3)
    y_nz_int4 = i8toi4(y_transpose_nz_int8)
    y_nz = i4toi8(y_nz_int4)
    gl = np.zeros([EXP], dtype=np.int32)
    out = np.zeros([M, N], dtype=np.float32)
    y_scale_fp16 = y_scale.astype(np.float16)
    y_int = y_int8.reshape(EXP, K//G, G, N)
    before = 0
    for i in range(EXP):
        gl[i] = random.randint(before, min(M, before + 1))
        for j in range(gl[i] - before):
            for k in range(K // G):
                out[before + j, :] += np.matmul(x_int8[before + j, k*G:(k+1)*G].astype(np.int32), y_int[i, k, :, :].reshape(
                    [G, N]).astype(np.int32)).astype(np.float32) * (y_scale_fp16[i, k, :, :].reshape((N)))
            out[before + j, :] = out[before + j, :] * x_scale[before + j, :]
        before = gl[i]

    x_int8.tofile(f"./temp/{OP_NAME}/input/gm_x.bin")
    y_nz.tofile(f"./temp/{OP_NAME}/input/gm_weight.bin")
    gl.tofile(f"./temp/{OP_NAME}/input/group_list.bin")
    y_scale_fp16.tofile(f"./temp/{OP_NAME}/input/gm_y_scale.bin")
    x_scale.tofile(f"./temp/{OP_NAME}/input/gm_x_scale.bin")
    out.astype(np.float16).tofile(f"./temp/{OP_NAME}/output/gm_out_golden.bin")


@sub_kernel(core_num=CORE_NUM)
def grouped_matmul(x, w, group_list, x_scale, y_scale, out):
    block_idx = get_block_idx()
    ub_list = slice_to_ub(group_list, [0], slicesize=[EXP])
    ub_tiling_size = 14
    ub_tiling = K // 32 // ub_tiling_size
    start_batch = Scalar("INT32", 0)
    l0b_tiling_size = 64
    l0b_tiling = ub_tiling_size * 32 // l0b_tiling_size

    for i in dynamic_loop(EXP):
        end_batch = move_to_scalar(ub_list[i])
        ub_out = vector_dup(Scalar("FP16", 0), [1, N//8], False)
        # [1, 512] @ [512, 7168] -> [1, 7168]
        for j in dynamic_loop(end_batch - start_batch):
            ub_scale_x = slice_to_ub(x_scale, [start_batch + j], [1])
            ub_scale_scalar = move_to_scalar(ub_scale_x)
            for n in dynamic_loop(ub_tiling):
                x_ub = slice_to_ub(
                    x, [start_batch + j, n * ub_tiling_size * 32], [1, ub_tiling_size * 32])
                x_ub_nz = change_view(x_ub, new_format="NZ")
                x_ub_pad = pad_to_ub(x_ub_nz, [16, ub_tiling_size * 32])
                w_ub = slice_to_ub(
                    w, [i, n * ub_tiling_size, block_idx, 0], [1, ub_tiling_size, 1, N * 32 // 8 // 4])
                w_ub_nd = change_view(w_ub, [1*ub_tiling_size*N], "ND")
                w_ub_int8 = vconv_s42s8(w_ub_nd)
                w_ub_nz = change_view(
                    w_ub_int8, [1, N//8, ub_tiling_size*32], "NZ")
                w_l1 = move_to_l1(w_ub_nz)
                x_l1 = move_to_l1(x_ub_pad)
                for k in range(l0b_tiling):
                    x_l0a = slice_to_l0A(
                        x_l1, [0, k * l0b_tiling_size], [16, l0b_tiling_size])
                    w_l0b = slice_to_l0B(
                        w_l1, [0, 0, k * l0b_tiling_size], [1, N//8, l0b_tiling_size], transpose=True)
                    l0c = mmad(x_l0a, w_l0b)
                    ub_y_scale = slice_to_ub(
                        y_scale, [i, n*l0b_tiling + k, 0, block_idx * N//8], [1, 1, 1, N//8])
                    ub_y_scale_cv = change_view(ub_y_scale, [1,  N//8])
                    ub_l0c = move_to_ub(l0c)
                    ub_l0c_f32 = vconv(ub_l0c, "FP32")
                    ub_l0c_quant = vmuls(ub_l0c_f32, ub_scale_scalar)
                    ub_l0c_fp16 = vconv(ub_l0c_quant, "FP16")
                    ub_l0c_nd = nz_to_nd(ub_l0c_fp16)
                    ub_l0c_slice = slice_to_ub(
                        ub_l0c_nd, [0, 0, 0], [1, 1, N//8])
                    ub_l0c_nd_1 = change_view(ub_l0c_slice, [1, N//8])
                    ub_out_mul = vmul(ub_l0c_nd_1, ub_y_scale_cv)
                    ub_out = vadd(ub_out, ub_out_mul)
            insert_to_gm(
                out, ub_out, [start_batch + j, block_idx * N//8], [1, N//8])

        start_batch = end_batch.copy()


if __name__ == "__main__":
    gen_data()
    gm_x = Tensor("GM", "INT8", [M, K], "ND", False)
    gm_weight = Tensor(
        "GM", "INT16", [EXP, K // 32, 8, N * 32 // 8 // 4], "ND", False)
    group_list = Tensor("GM", "INT32", [EXP], "ND", False)
    gm_x_scale = Tensor("GM", "FP32", [M], "NZ", False)
    gm_y_scale = Tensor("GM", "FP16", [EXP, K // G, 1, N], "ND", False)
    gm_out = Tensor("GM", "FP16", [M, N], "ND", False)
    compile_func(grouped_matmul, globals())(gm_x, gm_weight,
                                            group_list, gm_x_scale, gm_y_scale, gm_out)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'gm_x', 'gm_weight', 'group_list', 'gm_x_scale', 'gm_y_scale'], outputs=['gm_out'], profiling=100)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_out_actual.bin ./temp/{OP_NAME}/output/gm_out_golden.bin float16 4e-2 1e-2 4e-3')
    sys.exit(return_code >> 8)
