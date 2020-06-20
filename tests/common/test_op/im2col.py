# Copyright 2019 Huawei Technologies Co., Ltd
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

import tvm

def intrin_load3d(dtype):
    '''
    Create intrin function call for im2col

    Args:
        dtype: type of the data

    Return:
        cce intrin function call for im2col
    '''

    # fmatrix -------------------------------------------
    input_w = tvm.var('input_width')
    input_h = tvm.var('input_height')

    pad_left = tvm.var('padding_left')
    pad_right = tvm.var('padding_right')
    pad_top = tvm.var('padding_top')
    pad_bottom = tvm.var('padding_bottom')
    # ---------------------------------------------------
    # xm ------------------------------------------------
    w_idx_kernel = tvm.var('fetch_position_inside_kernel_width')
    h_idx_kernel = tvm.var('fetch_position_inside_kernel_height')

    h_idx = tvm.var('kernel_h_index_in_input')
    w_idx = tvm.var('kernel_w_index_in_input')

    c1_idx = tvm.var("position_c1")
    # ---------------------------------------------------
    # xt ------------------------------------------------
    stride_h = tvm.var('stride_h')
    stride_w = tvm.var('stride_w')

    kernel_h = tvm.var('kernel_h')
    kernel_w = tvm.var('kernel_w')

    dilation_h = tvm.var('dilation_h')
    dilation_w = tvm.var('dilation_h')

    jump_offset = tvm.var('jump_offset')
    repeat_mode = tvm.var('repeat_mode')
    repeat_time = tvm.var('repeat_time')
    # ---------------------------------------------------
    # csize ----------------------------------------
    csize = tvm.var('csize')
    # ---------------------------------------------------

    window_size = 16

    input_b  = 1
    input_c1 = 1
    input_c0 = 16

    input_data = tvm.placeholder((input_b, input_c1, input_h, input_w, input_c0), dtype=dtype)

    result = tvm.compute((window_size, input_c0),
                         lambda window, c0:
                            input_data[0,
                                       0,
                                       h_idx + h_idx_kernel + pad_bottom,
                                       w_idx + w_idx_kernel + pad_left + window*stride_w,
                                       c0],
                         name='img2col_intrinsic')

    input_data_buff = tvm.decl_buffer(input_data.shape, input_data.dtype,
                                      name="input_data_buff",
                                      offset_factor=1, scope="local.L1")

    result_buff = tvm.decl_buffer(result.shape, result.dtype,
                                  name="result_buff",
                                  offset_factor=1, scope="local.UB")
    def s2u(num, bits):
      return (num+2**bits)%(2**bits)

    def pack_args(sp):
        if not len(sp) == 21:
            raise ValueError("sp should be 21 while given sp is %i" % len(sp))
        # fmatrix = (sp[0] & 0xffff) << 0 | (sp[1] & 0xffff) << 16 | (sp[2] & 0xff) << 32 | (sp[3] & 0xff) << 40 | (sp[4] & 0xff) << 48 | (sp[5] & 0xff) << 56
        # xm = (sp[6] & 0xff) << 16 | (sp[7] & 0xff) << 24 | (sp[8] & 0xffff) << 32 | (sp[9] & 0xffff) << 48 | (sp[10] & 0xfff) << 0
        # xt = (sp[11] & 63) << 0 | (sp[12] & 63) << 6 | (sp[13] & 0xff) << 12 | (sp[14] & 0xff) << 20 | (sp[15] & 0xff) << 28 | (sp[16] & 0xff) << 36 | (sp[17] & 0xff) << 44 | (sp[18] & 1) << 52 | (sp[19] & 0xff) << 56
        fmatrix = sp[0] + sp[1]  * tvm.const(2**16, 'uint64') + sp[2] * tvm.const(2**32, 'uint64') + sp[3] * tvm.const(2**40, 'uint64') + sp[4] * tvm.const(2**48, 'uint64') + sp[5] * tvm.const(2**56, 'uint64')
        xm = sp[6] * tvm.const(2**16, 'uint64') + sp[7] * tvm.const(2**24, 'uint64') + s2u(sp[8], 16) * tvm.const(2**32, 'uint64') + s2u(sp[9], 16) * tvm.const(2**48, 'uint64') + sp[10]
        xt = sp[11] + sp[12] * tvm.const(2**6, 'uint64') + sp[13] * tvm.const(2**12, 'uint64') + sp[14] * tvm.const(2**20, 'uint64') + sp[15] * tvm.const(2**28, 'uint64') + sp[16] * tvm.const(2**36, 'uint64') + sp[17] * tvm.const(2**44, 'uint64') + sp[18] * tvm.const(2**52, 'uint64') + sp[19] * tvm.const(2**56, 'uint64')
        csize = sp[20]

        return (fmatrix, xm, xt, csize)


    def intrin_func(ins, outs, sp):
        aa = ins[0]
        bb = outs[0]
        ib = tvm.ir_builder.create()
        fmatrix, xm, xt, csize = pack_args(sp)
        ib.emit(tvm.call_extern(dtype, "set_fmatrix", fmatrix))
        ib.emit(tvm.call_extern(dtype, "img2col_cbuf_to_ub",
                                bb.access_ptr("w"),
                                aa.access_ptr("r"),
                                xm, xt, csize))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(result.op,
                                      intrin_func,
                                      binds={input_data: input_data_buff, result: result_buff},
                                      scalar_params=[input_w, input_h, pad_left, pad_right, pad_top, pad_bottom,  # fmatrix
                                                     w_idx_kernel, h_idx_kernel, w_idx, h_idx, c1_idx,  # xm
                                                     stride_w, stride_h, kernel_w, kernel_h, dilation_w, dilation_h, jump_offset, repeat_mode, repeat_time,  # xt
                                                     csize])

