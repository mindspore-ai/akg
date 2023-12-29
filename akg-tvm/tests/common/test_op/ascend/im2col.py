# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

import akg.tvm

BLOCK_SIZE = 16


def intrin_load_im2col(dtype, stride_w, stride_h, kernel_w, kernel_h, dilation_h=1, dilation_w=1):
    '''
    Create intrin function call for im2col

    Args:
        dtype: type of the data

    Return:
        cce intrin function call for im2col
    '''
    input_w = akg.tvm.var('input_width')
    input_h = akg.tvm.var('input_height')
    pad_left = akg.tvm.var('padding_left')
    pad_right = akg.tvm.var('padding_right')
    pad_top = akg.tvm.var('padding_top')
    pad_bottom = akg.tvm.var('padding_bottom')
    w_idx_kernel = akg.tvm.var('fetch_position_inside_kernel_width')
    h_idx_kernel = akg.tvm.var('fetch_position_inside_kernel_height')
    h_idx = akg.tvm.var('kernel_h_index_in_input')
    w_idx = akg.tvm.var('kernel_w_index_in_input')

    window_size = 16
    input_b = 1
    input_c1 = 1
    input_c0 = 16
    input_data = akg.tvm.placeholder(
        (input_b, input_c1, input_h, input_w, input_c0), dtype=dtype)
    result = akg.tvm.compute((window_size, input_c0),
                             lambda window, c0:
                             input_data[0,
                                        0,
                                        h_idx + h_idx_kernel + pad_bottom,
                                        w_idx + w_idx_kernel + pad_left + window*stride_w,
                                        c0],
                             name='img2col_intrinsic')
    input_data_buff = akg.tvm.decl_buffer(input_data.shape, input_data.dtype,
                                          name="input_data_buff",
                                          offset_factor=1, scope="local.L1")
    result_buff = akg.tvm.decl_buffer(result.shape, result.dtype,
                                      name="result_buff",
                                      offset_factor=1, scope="local.UB")

    def intrin_func(ins, outs, sp):
        c1_idx = 0
        offset = 1
        mode = 0
        time = 1
        csize = 0
        aa = ins[0]
        bb = outs[0]
        ib = akg.tvm.ir_builder.create()
        call_args = [sp[0], sp[1], sp[2], sp[3], sp[4], sp[5],
                     sp[6], sp[7], sp[8], sp[9], c1_idx,
                     stride_w, stride_h, kernel_w, kernel_h, dilation_w, dilation_h,
                     offset, mode, time,
                     csize]
        ib.emit(akg.tvm.call_extern(dtype, "img2col_cbuf_to_ub",
                                    bb.access_ptr("w"),
                                    aa.access_ptr("r"),
                                    *call_args))
        return ib.get()

    with akg.tvm.build_config(offset_factor=1):
        return akg.tvm.decl_tensor_intrin(result.op,
                                          intrin_func,
                                          binds={
                                              input_data: input_data_buff, result: result_buff},
                                          scalar_params=[input_w, input_h, pad_left, pad_right, pad_top, pad_bottom,
                                                         w_idx_kernel, h_idx_kernel, w_idx, h_idx])


def im2col_manual_schedule(data, kernel, stride, pad, target="cce"):
    '''
    Compute im2col via cce im2col intrin function call directly

    Args:
        data (akg.tvm.tensor.Tensor): Tensor of type float16, float32.
        kernel (Union[list, tuple]): List or tuple of two int numbers for pooling window's size.
        stride (Union[list, tuple]): List or tuple of two int numbers for window's stride.
        pad (Union[List, tuple]): List or tuple of four int numbers for padding(top, bottom, left, and right).

    Return:
        akg.tvm.tensor.Tensor of same type as data, shape is the zN?.
    '''

    b, c1, h, w, c0 = data.shape
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel
    pad_t, pad_b, pad_l, pad_r = pad
    # output size <=> number of windows
    ho = (h + pad_b + pad_t - kernel_h) // stride_h + 1
    wo = (w + pad_r + pad_l - kernel_w) // stride_w + 1
    load_im2col = intrin_load_im2col(
        data.dtype, stride_w, stride_h, kernel_w, kernel_h)
    im2col_shape = (b,
                    (ho * wo + BLOCK_SIZE - 1) // BLOCK_SIZE,
                    c1 * kernel_h * kernel_w,
                    BLOCK_SIZE,
                    c0)

    def _im2col_compute(i, j, k, data):
        j_h = (((j * BLOCK_SIZE) // wo) * stride_h) - pad_t
        j_w = (((j * BLOCK_SIZE) % wo) * stride_w) - pad_l
        h_3d = kernel_h - akg.tvm.max(((j_h + kernel_h) - h), 0)
        pad_t_3d = akg.tvm.max(-j_h, 0)
        pad_b_3d = akg.tvm.max(((j_h + kernel_h) - h), 0)
        w_idx_kernel = (k % kernel_w)
        h_idx_kernel = ((k // kernel_w) % kernel_h)
        w_idx = j_w
        h_idx = akg.tvm.min(j_h, 0)
        c1_idx = (k // kernel_w) // kernel_h

        load_im2col_input = data[i,
                                 c1_idx,
                                 akg.tvm.max(j_h, 0):akg.tvm.min(j_h + kernel_h, h),
                                 0:w,
                                 0:c0]

        return load_im2col(load_im2col_input,
                           w, h_3d, pad_l, pad_r, pad_t_3d, pad_b_3d,
                           w_idx_kernel, h_idx_kernel, w_idx, h_idx)

    # assume we need the whole width of a
    # choose a section of the rows of a that encompasses all of the windows in the current window-batch
    res = akg.tvm.compute(im2col_shape,
                          lambda i, j, k:
                          _im2col_compute(i, j, k, data),
                          name='im2col_fractal')

    def comp_func(s):
        data_l1 = s.cache_read(data, "local.L1", [res])
        res_ub = s.cache_write(res, "local.UB")

        b_ax, hw1_ax, c1_kh_kw_ax, hw0_ax, c0_ax = res.op.axis
        hw1_out = hw1_ax
        if akg.tvm.all([wo > BLOCK_SIZE]):
            cut_w1 = wo // BLOCK_SIZE
            cut_h1 = 1
            cut_hw1 = cut_h1 * cut_w1
            hw1_out, hw1_in = s[res].split(hw1_ax, cut_hw1)
        s[data_l1].compute_at(s[res], hw1_out)
        s[res_ub].compute_at(s[res], c1_kh_kw_ax)

    return res, comp_func
