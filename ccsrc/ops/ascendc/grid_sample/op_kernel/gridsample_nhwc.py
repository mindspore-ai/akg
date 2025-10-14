import numpy as np
import os
import sys
from swft.core import *
from swft.api import *

OP_NAME = 'grid_sample'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

N = 1
C = 1536 # C should be 8 aligned
H_IN = 24
W_IN = 24
H_OUT = 10764
W_OUT= 1

PER_LOOP_PIXEL_NUM = 8
BLOCK_LOOP_NUM = 8 # 32Bytes / sizeof(type)
CORE_NUM = 8
ALIGN_CORNERS = False
PADDING_MODE = 1
CHANNEL_LOOP_CHUNK = 512 # C should be 512 aligned
CHANNEL_LOOP_CHUNK2 = 8 # chunk size for residual
INTERPOLATION_MODE = 0

# Numpy Test
# ===============================================================================
def bilinear_interpolate(input_tensor, x, y, H, W):
    """双线性插值"""
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = min(x1 + 1, W - 1)
    y2 = min(y1 + 1, H - 1)

    # 边界检查
    if x1 < 0 or x1 >= W or y1 < 0 or y1 >= H:
        return np.zeros(input_tensor.shape[0], dtype=np.float32)

    # 计算权重
    wx = x - x1
    wy = y - y1

    # 双线性插值
    result = (input_tensor[y1, x1, :] * (1 - wx) * (1 - wy) +
              input_tensor[y1, x2, :] * wx * (1 - wy) +
              input_tensor[y2, x1, :] * (1 - wx) * wy +
              input_tensor[y2, x2, :] * wx * wy)

    return result

def gen_data():
    np.random.seed(0)
    # input_data = np.arange(N * H_IN * W_IN * C).reshape(N, H_IN, W_IN, C).astype(np.float32)
    # input_data = np.full([N, H_IN, W_IN, C], 100.0, dtype=np.float32)
    # input_data = np.repeat(np.arange(1, N * H_IN * W_IN + 1).reshape(N, H_IN, W_IN, 1), C, axis=3).astype(np.float32)
    input_data = np.random.uniform(-1, 1, [N, H_IN, W_IN, C]).astype(np.float32)
    grid_data = np.random.uniform(-1, 1, [N, H_OUT, W_OUT, 2]).astype(np.float32)
    output_data = np.zeros([N, H_OUT, W_OUT, C]).astype(np.float32)

    for n in range(N):
        for h in range(H_OUT):
            for w in range(W_OUT):
                # 获取归一化坐标
                x_norm = grid_data[n, h, w, 0]
                y_norm = grid_data[n, h, w, 1]

                # 映射到input坐标空间
                if ALIGN_CORNERS:
                    x = (x_norm + 1) * (W_IN - 1) / 2
                    y = (y_norm + 1) * (H_IN - 1) / 2
                else:
                    x = (x_norm + 1) * W_IN / 2 - 0.5
                    y = (y_norm + 1) * H_IN / 2 - 0.5

                # 边界处理
                if PADDING_MODE == 0:  # zeros
                    if x < 0 or x >= W_IN or y < 0 or y >= H_IN:
                        output_data[n, h, w, :] = 0
                        continue
                elif PADDING_MODE == 1:  # border
                    x = np.clip(x, 0, W_IN - 1)
                    y = np.clip(y, 0, H_IN - 1)

                # 插值采样
                if INTERPOLATION_MODE == 0:  # bilinear
                    output_data[n, h, w, :] = bilinear_interpolate(
                        input_data[n], x, y, H_IN, W_IN)
                elif INTERPOLATION_MODE == 1:  # nearest
                    x_idx = int(x)
                    y_idx = int(y)
                    x_idx = np.clip(x_idx, 0, W_IN - 1)
                    y_idx = np.clip(y_idx, 0, H_IN - 1)
                    output_data[n, h, w, :] = input_data[n, y_idx, x_idx, :]

    h_in = np.array([H_IN], dtype=np.float32)
    w_in = np.array([W_IN], dtype=np.float32)
    h_out = np.array([H_OUT], dtype=np.int32)
    w_out = np.array([W_OUT], dtype=np.int32)
    n_in = np.array([N], dtype=np.int32)
    c_in = np.array([C], dtype=np.int32)

    # 保存测试数据
    h_in.tofile(f"./temp/{OP_NAME}/input/h_in.bin")
    w_in.tofile(f"./temp/{OP_NAME}/input/w_in.bin")
    h_out.tofile(f"./temp/{OP_NAME}/input/h_out.bin")
    w_out.tofile(f"./temp/{OP_NAME}/input/w_out.bin")
    n_in.tofile(f"./temp/{OP_NAME}/input/n_in.bin")
    c_in.tofile(f"./temp/{OP_NAME}/input/c_in.bin")
    input_data.tofile(f"./temp/{OP_NAME}/input/input_data.bin")
    grid_data.tofile(f"./temp/{OP_NAME}/input/grid_data.bin")
    output_data.tofile(f"./temp/{OP_NAME}/output/output_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=CORE_NUM)
def grid_sample(input_data, grid_data, output_data, h_in, w_in, h_out, w_out, n_in, c_in):
    block_idx = get_block_idx()

    # calc number of output this rank need to compute
    total_outputs = n_in * h_out * w_out
    # 将total_outputs按照 Block_Loop_NUM 分组
    block_num = total_outputs // BLOCK_LOOP_NUM
    # 多余的 outputs
    output_res = total_outputs % BLOCK_LOOP_NUM
    percore_blocks = Scalar("INT32", 0)
    percore_blocks = (block_num + CORE_NUM - 1) // CORE_NUM
    pixel_num = Scalar("INT32", 0)
    if (block_idx + 1) * percore_blocks < block_num:
        pixel_num = percore_blocks * BLOCK_LOOP_NUM
    elif block_idx * percore_blocks < block_num:
        pixel_num = (block_num - block_idx * percore_blocks) * BLOCK_LOOP_NUM + output_res
    else:
        pixel_num = Scalar("INT32", 0)
    start_idx = block_idx * percore_blocks * BLOCK_LOOP_NUM
    if (block_idx == 0) and pixel_num < 1:
        pixel_num = total_outputs
    
    max_pixel_num = PER_LOOP_PIXEL_NUM # 32Byte aligned
    tile_num = pixel_num // max_pixel_num
    res_num = pixel_num % max_pixel_num

    c_chunk_num = c_in // CHANNEL_LOOP_CHUNK
    c_res_num = c_in % CHANNEL_LOOP_CHUNK
    
    # C should be 8 aligned
    c_res_chunk_num = (c_res_num + CHANNEL_LOOP_CHUNK2 - 1) // CHANNEL_LOOP_CHUNK2
    c_res_start = c_chunk_num * CHANNEL_LOOP_CHUNK

    in_stride_2 = Scalar("INT32", 0)
    in_stride_2 = c_in
    in_stride_1 = w_in.astype("INT32") * in_stride_2
    in_stride_0 = h_in.astype("INT32") * in_stride_1

    out_stride_2 = Scalar("INT32", 0)
    out_stride_2 = c_in
    out_stride_1 = w_out * out_stride_2
    out_stride_0 = h_out * out_stride_1

    # consts
    const1 = w_in * 0.5
    const2 = h_in * 0.5
    const3 = const1 - 0.5
    const4 = const2 - 0.5
    const5 = w_in - 1
    const6 = h_in - 1
    const7 = const5.astype("INT32")
    const8 = const6.astype("INT32")
    for i in dynamic_loop(tile_num):
        idx = start_idx + i * max_pixel_num
        xys_ub = slice_to_ub(grid_data, [idx, 0], [max_pixel_num, 2])
        xys_ub_trans = transpose(xys_ub, [1, 0])
        xs_ub = slice_to_ub(xys_ub_trans, [0, 0], [1, max_pixel_num])
        ys_ub = slice_to_ub(xys_ub_trans, [1, 0], [1, max_pixel_num])

        # align corners is false
        xs_ub = vmuls(xs_ub, const1)
        ys_ub = vmuls(ys_ub, const2)
        xs_ub = vadds(xs_ub, const3)
        ys_ub = vadds(ys_ub, const4)

        # border clip
        xs_ub = vmaxs(xs_ub, Scalar("FP32", 0))
        ys_ub = vmaxs(ys_ub, Scalar("FP32", 0))
        xs_ub = vmins(xs_ub, const5)
        ys_ub = vmins(ys_ub, const6)
        
        # bilinear interpolation
        xs_ub_w_int = vconv(xs_ub, "INT32", "z")
        xs_ub_w = vconv(xs_ub_w_int, "FP32")
        xs_ub_e_int = vadds(xs_ub_w_int, Scalar("INT32", 1))
        xs_ub_e_int = vmins(xs_ub_e_int, const7)
        xs_ub_e = vadds(xs_ub_w, Scalar("FP32", 1))

        ys_ub_n_int = vconv(ys_ub, "INT32", "z")
        ys_ub_n = vconv(ys_ub_n_int, "FP32")
        ys_ub_s_int = vadds(ys_ub_n_int, Scalar("INT32", 1))
        ys_ub_s_int = vmins(ys_ub_s_int, const8)
        ys_ub_s = vadds(ys_ub_n, Scalar("FP32", 1))
        
        w_weights = vsub(xs_ub_e, xs_ub)
        e_weights = vsub(xs_ub, xs_ub_w)
        n_weights = vsub(ys_ub_s, ys_ub)
        s_weights = vsub(ys_ub, ys_ub_n)

        nw_weights = vmul(w_weights, n_weights)
        ne_weights = vmul(e_weights, n_weights)
        sw_weights = vmul(w_weights, s_weights)
        se_weights = vmul(e_weights, s_weights)
        
        # n/e may be out of range, but weight will be zero
        for j in dynamic_loop(max_pixel_num):
            inner_idx = (idx + j)
            hw = inner_idx % (h_out * w_out)
            w = hw % w_out
            h = hw // w_out
            n = inner_idx // (h_out * w_out)

            x_w = move_to_scalar(xs_ub_w_int[0, j])
            x_e = move_to_scalar(xs_ub_e_int[0, j])
            y_n = move_to_scalar(ys_ub_n_int[0, j])
            y_s = move_to_scalar(ys_ub_s_int[0, j])
            
            nw_weight = move_to_scalar(nw_weights[0, j])
            ne_weight = move_to_scalar(ne_weights[0, j])
            sw_weight = move_to_scalar(sw_weights[0, j])
            se_weight = move_to_scalar(se_weights[0, j])
            
            
            for idx_c in dynamic_loop(c_chunk_num):
                input_ub_nw = slice_to_ub(input_data, [n * in_stride_0 + y_n * in_stride_1 + x_w * in_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                                          [CHANNEL_LOOP_CHUNK])
                input_ub_ne = slice_to_ub(input_data, [n * in_stride_0 + y_n * in_stride_1 + x_e * in_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                                          [CHANNEL_LOOP_CHUNK])
                input_ub_sw = slice_to_ub(input_data, [n * in_stride_0 + y_s * in_stride_1 + x_w * in_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                                          [CHANNEL_LOOP_CHUNK])
                input_ub_se = slice_to_ub(input_data, [n * in_stride_0 + y_s * in_stride_1 + x_e * in_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                                          [CHANNEL_LOOP_CHUNK])
                
                input_ub = vmuls(input_ub_nw, nw_weight)
                input_ub = vadd(input_ub, vmuls(input_ub_ne, ne_weight))
                input_ub = vadd(input_ub, vmuls(input_ub_sw, sw_weight))
                input_ub = vadd(input_ub, vmuls(input_ub_se, se_weight))

                insert_to_gm(output_data, input_ub, [n * out_stride_0 + h * out_stride_1 + w * out_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                             [CHANNEL_LOOP_CHUNK])

            for idx_c in dynamic_loop(c_res_chunk_num):
                input_ub_nw = slice_to_ub(input_data, [n * in_stride_0 + y_n * in_stride_1 + x_w * in_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                                          [CHANNEL_LOOP_CHUNK2])
                input_ub_ne = slice_to_ub(input_data, [n * in_stride_0 + y_n * in_stride_1 + x_e * in_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                                          [CHANNEL_LOOP_CHUNK2])
                input_ub_sw = slice_to_ub(input_data, [n * in_stride_0 + y_s * in_stride_1 + x_w * in_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                                          [CHANNEL_LOOP_CHUNK2])
                input_ub_se = slice_to_ub(input_data, [n * in_stride_0 + y_s * in_stride_1 + x_e * in_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                                          [CHANNEL_LOOP_CHUNK2])

                input_ub = vmuls(input_ub_nw, nw_weight)
                input_ub = vadd(input_ub, vmuls(input_ub_ne, ne_weight))
                input_ub = vadd(input_ub, vmuls(input_ub_sw, sw_weight))
                input_ub = vadd(input_ub, vmuls(input_ub_se, se_weight))

                insert_to_gm(output_data, input_ub, [n * out_stride_0 + h * out_stride_1 + w * out_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                             [CHANNEL_LOOP_CHUNK2])

    # for residual
    if res_num > 0:
        idx_ = start_idx + pixel_num - res_num
        xys_ub_ = slice_to_ub(grid_data, [idx_, 0], [max_pixel_num, 2])
        xys_ub_trans_ = transpose(xys_ub_, [1, 0])
        xs_ub_ = slice_to_ub(xys_ub_trans_, [0, 0], [1, max_pixel_num])
        ys_ub_ = slice_to_ub(xys_ub_trans_, [1, 0], [1, max_pixel_num])

        # align corners is false
        xs_ub_ = vmuls(xs_ub_, const1)
        ys_ub_ = vmuls(ys_ub_, const2)
        xs_ub_ = vadds(xs_ub_, const3)
        ys_ub_ = vadds(ys_ub_, const4)

        # border clip
        xs_ub_ = vmaxs(xs_ub_, Scalar("FP32", 0))
        ys_ub_ = vmaxs(ys_ub_, Scalar("FP32", 0))
        xs_ub_ = vmins(xs_ub_, const5)
        ys_ub_ = vmins(ys_ub_, const6)
        
        # bilinear interpolation
        xs_ub_w_int_ = vconv(xs_ub_, "INT32", "z")
        xs_ub_w_ = vconv(xs_ub_w_int_, "FP32")
        xs_ub_e_int_ = vadds(xs_ub_w_int_, Scalar("INT32", 1))
        xs_ub_e_int_ = vmins(xs_ub_e_int_, const7)
        xs_ub_e_ = vadds(xs_ub_w_, Scalar("FP32", 1))

        ys_ub_n_int_ = vconv(ys_ub_, "INT32", "z")
        ys_ub_n_ = vconv(ys_ub_n_int_, "FP32")
        ys_ub_s_int_ = vadds(ys_ub_n_int_, Scalar("INT32", 1))
        ys_ub_s_int_ = vmins(ys_ub_s_int_, const8)
        ys_ub_s_ = vadds(ys_ub_n_, Scalar("FP32", 1))
        
        w_weight_ = vsub(xs_ub_e_, xs_ub_)
        e_weight_ = vsub(xs_ub_, xs_ub_w_)
        n_weight_ = vsub(ys_ub_s_, ys_ub_)
        s_weight_ = vsub(ys_ub_, ys_ub_n_)
        
        nw_weight_ = vmul(w_weight_, n_weight_)
        ne_weight_ = vmul(e_weight_, n_weight_)
        sw_weight_ = vmul(w_weight_, s_weight_)
        se_weight_ = vmul(e_weight_, s_weight_)
        
        # n/e may be out of range, but weight will be zero

        for j in dynamic_loop(res_num):
            inner_idx = (idx_ + j)
            hw = inner_idx % (h_out * w_out)
            w = hw % w_out
            h = hw // w_out
            n = inner_idx // (h_out * w_out)
            
            x_w = move_to_scalar(xs_ub_w_int_[0, j])
            x_e = move_to_scalar(xs_ub_e_int_[0, j])
            y_n = move_to_scalar(ys_ub_n_int_[0, j])
            y_s = move_to_scalar(ys_ub_s_int_[0, j])
            
            nw_weight = move_to_scalar(nw_weight_[0, j])
            ne_weight = move_to_scalar(ne_weight_[0, j])
            sw_weight = move_to_scalar(sw_weight_[0, j])
            se_weight = move_to_scalar(se_weight_[0, j])
            
            for idx_c in dynamic_loop(c_chunk_num):
                input_ub_nw = slice_to_ub(input_data, [n * in_stride_0 + y_n * in_stride_1 + x_w * in_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                                          [CHANNEL_LOOP_CHUNK])
                input_ub_ne = slice_to_ub(input_data, [n * in_stride_0 + y_n * in_stride_1 + x_e * in_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                                          [CHANNEL_LOOP_CHUNK])
                input_ub_sw = slice_to_ub(input_data, [n * in_stride_0 + y_s * in_stride_1 + x_w * in_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                                          [CHANNEL_LOOP_CHUNK])
                input_ub_se = slice_to_ub(input_data, [n * in_stride_0 + y_s * in_stride_1 + x_e * in_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                                          [CHANNEL_LOOP_CHUNK])

                input_ub = vmuls(input_ub_nw, nw_weight)
                input_ub = vadd(input_ub, vmuls(input_ub_ne, ne_weight))
                input_ub = vadd(input_ub, vmuls(input_ub_sw, sw_weight))
                input_ub = vadd(input_ub, vmuls(input_ub_se, se_weight))

                insert_to_gm(output_data, input_ub, [n * out_stride_0 + h * out_stride_1 + w * out_stride_2 + idx_c * CHANNEL_LOOP_CHUNK],
                             [CHANNEL_LOOP_CHUNK])
            
            for idx_c in dynamic_loop(c_res_chunk_num):
                input_ub_nw = slice_to_ub(input_data, [n * in_stride_0 + y_n * in_stride_1 + x_w * in_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                                          [CHANNEL_LOOP_CHUNK2])
                input_ub_ne = slice_to_ub(input_data, [n * in_stride_0 + y_n * in_stride_1 + x_e * in_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                                          [CHANNEL_LOOP_CHUNK2])
                input_ub_sw = slice_to_ub(input_data, [n * in_stride_0 + y_s * in_stride_1 + x_w * in_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                                          [CHANNEL_LOOP_CHUNK2])
                input_ub_se = slice_to_ub(input_data, [n * in_stride_0 + y_s * in_stride_1 + x_e * in_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                                          [CHANNEL_LOOP_CHUNK2])

                input_ub = vmuls(input_ub_nw, nw_weight)
                input_ub = vadd(input_ub, vmuls(input_ub_ne, ne_weight))
                input_ub = vadd(input_ub, vmuls(input_ub_sw, sw_weight))
                input_ub = vadd(input_ub, vmuls(input_ub_se, se_weight))

                insert_to_gm(output_data, input_ub, [n * out_stride_0 + h * out_stride_1 + w * out_stride_2 + c_res_start + idx_c * CHANNEL_LOOP_CHUNK2],
                             [CHANNEL_LOOP_CHUNK2])


if __name__ == "__main__":
    set_context("310P")
    gen_data()
    input_data = Tensor("GM", "FP32", [N * H_IN * W_IN * C], "ND", False)
    grid_data = Tensor("GM", "FP32", [N * H_OUT * W_OUT, 2], "ND", False)
    output_data = Tensor("GM", "FP32", [N * H_OUT * W_OUT * C], "ND", False)
    h_in = Scalar("FP32")
    w_in = Scalar("FP32")
    h_out = Scalar("INT32")
    w_out = Scalar("INT32")
    n_in = Scalar("INT32")
    c_in = Scalar("INT32")
    compile_func(grid_sample, globals())(input_data, grid_data, output_data, h_in, w_in, h_out, w_out, n_in, c_in)
    # compile_func(grid_sample_floor, globals())(input_data, grid_data, output_data, h_in, w_in, h_out, w_out, n_in, c_in)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'input_data', 'grid_data', 'h_in', 'w_in', 'h_out', 'w_out', 'n_in', 'c_in'],
                outputs=['output_data'], device_id=1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code_1 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/output_data_actual.bin ./temp/{OP_NAME}/output/output_golden.bin float32 4e-2 1e-2 4e-3')
    sys.exit(return_code_1 >> 8)
