#!/usr/bin/env python3
# coding: utf-8
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

"""operator dsl function: maxpool_grad_with_argmax"""
import akg.tvm
import akg.topi
import akg
from akg.ops.nn.ascend.avgpool import cal_pad_shapes_by_strategy
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          (list, tuple), (list, tuple), (list, tuple), (str, list, tuple), (str, type(None)))
def MaxpoolGradWithArgmax(head, mask, shape, kernel, stride, pad, target=utils.CCE):
    """
    Automatic differentiate of maxpool with manual schedule.

     Args:
        head (tvm.tensor.Tensor): Tensor, the gradient needed to be propagation.
        mask (tvm.tensor.Tensor): Tensor, the mask indicating where the gradient is propagation.
        shape (Union[list, tuple]): five int numbers for pooling input image's size.
        kernel (Union[list, tuple]): two int numbers for pooling window's size.
        stride (Union[list, tuple]): two int numbers for window's stride.
        pad (Union[str, list, tuple]): padding, should be 'VALID','SAME' or
            instance of list(four int numbers, as 'CONSTANTS' strategy).
            Support **pad** is the same as avgpool's **Strategies**.

    Returns:
        tvm.tensor.Tensor as result for gradient of maxpooling.
    
    Supported Platforms:
        'Ascend'
    """
    dtype = head.dtype

    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    [ph_h, _, pw_h, _], [out_size_h, out_size_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, pad)
    batch_size, input_c1, input_h, input_w, input_c0 = shape

    # tile size 14 by 14 is proved to be the most efficient one
    tile_scale_h = 7
    tile_scale_w = 7

    tile_h = stride_h * tile_scale_h

    if kernel_h == stride_h:  # non-overlapping case
        tile_h_pad_u = ph_h % stride_h
    elif kernel_h % stride_h == 0:
        tile_h_pad_u = kernel_h - stride_h - ph_h
    else:
        tile_h_pad_u = kernel_h - kernel_h % stride_h - ph_h
    tile_h_pad_l = kernel_h - stride_h + ph_h
    tile_input_h = tile_h + tile_h_pad_u + tile_h_pad_l
    tile_h_out = (input_h - 1) // tile_h + 1

    if ph_h % stride_h == 0:
        pad_output_h = ph_h // stride_h
    else:
        pad_output_h = ph_h // stride_h + 1

    if tile_h_pad_u % stride_h == 0:
        pad_output_h -= tile_h_pad_u // stride_h
    else:
        pad_output_h -= tile_h_pad_u // stride_h + 1

    tile_output_h = (tile_input_h - kernel_h) // stride_h + 1

    tile_w = stride_w * tile_scale_w
    if kernel_w == stride_w:  # non-overlapping case
        tile_w_pad_u = pw_h % stride_w
    elif kernel_w % stride_w == 0:
        tile_w_pad_u = kernel_w - stride_w - pw_h
    else:
        tile_w_pad_u = kernel_w - kernel_w % stride_w - pw_h
    tile_w_pad_l = kernel_w - stride_w + pw_h
    tile_input_w = tile_w + tile_w_pad_u + tile_w_pad_l
    tile_w_out = (input_w - 1) // tile_w + 1

    if pw_h % stride_w == 0:
        pad_output_w = pw_h // stride_w
    else:
        pad_output_w = pw_h // stride_w + 1

    if tile_w_pad_u % stride_w == 0:
        pad_output_w -= tile_w_pad_u // stride_w
    else:
        pad_output_w -= tile_w_pad_u // stride_w + 1

    tile_output_w = (tile_input_w - kernel_w) // stride_w + 1

    cce_col2img = intrin_col2im((tile_h, tile_w),
                                (tile_output_h, tile_output_w),
                                kernel, stride,
                                (tile_h_pad_u, tile_h_pad_l, tile_h_pad_u, tile_h_pad_l),
                                "float32")

    head_reshaped = akg.tvm.compute((batch_size, input_c1, tile_h_out, tile_w_out,
                                     tile_output_h, tile_output_w, input_c0),
                                    lambda b, c1, h_out, w_out, oh, ow, c0:
                                    akg.tvm.expr.Select(
                                        akg.tvm.any(h_out * tile_scale_h + pad_output_h + oh < 0,
                                                    h_out * tile_scale_h + pad_output_h + oh > out_size_h - 1,
                                                    w_out * tile_scale_w + pad_output_w + ow < 0,
                                                    w_out * tile_scale_w + pad_output_w + ow > out_size_w - 1),
                                        akg.tvm.const(0.0, dtype=dtype),
                                        head(b, c1,
                                             h_out * tile_scale_h + pad_output_h + oh,
                                             w_out * tile_scale_w + pad_output_w + ow,
                                             c0)),
                                    name="head_reshaped")

    mask_reshaped = akg.tvm.compute((batch_size, input_c1, tile_h_out, tile_w_out, kernel_h, kernel_w,
                                     tile_output_h, tile_output_w, input_c0),
                                    lambda b, c1, h_out, w_out, kh, kw, oh, ow, c0:
                                    akg.tvm.expr.Select(
                                        akg.tvm.any(h_out * tile_scale_h + pad_output_h + oh < 0,
                                                    h_out * tile_scale_h + pad_output_h + oh > out_size_h - 1,
                                                    w_out * tile_scale_w + pad_output_w + ow < 0,
                                                    w_out * tile_scale_w + pad_output_w + ow > out_size_w - 1),
                                        akg.tvm.const(0.0, dtype=dtype),
                                        mask(b, c1, kh, kw,
                                             h_out * tile_scale_h + pad_output_h + oh,
                                             w_out * tile_scale_w + pad_output_w + ow,
                                             c0)),
                                    name="mask_reshaped")

    d_data = akg.tvm.compute((batch_size, input_c1, tile_h_out, tile_w_out, kernel_h, kernel_w,
                              tile_output_h, tile_output_w, input_c0),
                             lambda b, c1, h_out, w_out, kh, kw, oh, ow, c0:
                             mask_reshaped(b, c1, h_out, w_out, kh, kw, oh, ow, c0)
                             * head_reshaped(b, c1, h_out, w_out, oh, ow, c0),
                             name="d_data")

    d_data_cast = akg.tvm.compute(d_data.shape,
                                  lambda *i: d_data(*i).astype("float32"),
                                  name="d_data_cast.local.UB")

    result_tile = akg.tvm.compute((batch_size, input_c1, tile_h_out, tile_w_out,
                                   tile_h, tile_w, input_c0),
                                  lambda b, c1, h_out, w_out:
                                  cce_col2img(d_data_cast[b, c1, h_out, w_out,
                                                          0:kernel_h, 0:kernel_w,
                                                          0:tile_output_h, 0:tile_output_w,
                                                          0:input_c0]),
                                  name="result_tile.local.UB")

    result_cast = akg.topi.cast(result_tile, dtype)

    result = akg.tvm.compute(shape,
                             lambda b, c1, h, w, c0:
                             result_cast(b, c1, h // tile_h, w // tile_w, h % tile_h, w % tile_w, c0),
                             name="result")

    def comp_func(s):

        data_ub = s.cache_read(mask, "local.UB", [mask_reshaped])
        head_ub = s.cache_read(head, "local.UB", [head_reshaped])
        result_ub = s.cache_write(result, "local.UB")

        s[mask_reshaped].set_scope("local.UB")
        s[head_reshaped].set_scope("local.UB")
        s[d_data].set_scope("local.UB")
        s[d_data_cast].set_scope("local.UB")
        s[result_tile].set_scope("local.UB")
        s[result_cast].set_scope("local.UB")

        # inline output
        s[result_ub].compute_inline()

        # inline inputs
        s[head_ub].compute_inline()
        s[data_ub].compute_inline()

        # result_tile dependencies
        s[d_data_cast].compute_at(s[result_tile], result_tile.op.axis[3])
        s[d_data].compute_at(s[result_tile], result_tile.op.axis[3])
        s[mask_reshaped].compute_at(s[result_tile], result_tile.op.axis[3])
        s[head_reshaped].compute_at(s[result_tile], result_tile.op.axis[3])

        # tile result
        b, c1, h, w, c0 = result.op.axis
        h_out, h_in = s[result].split(h, tile_h)
        w_out, w_in = s[result].split(w, tile_w)
        s[result].reorder(b, c1, h_out, w_out, h_in, w_in, c0)
        s[result_tile].compute_at(s[result], w_out)
        s[result_cast].compute_at(s[result], w_out)

    return result, comp_func

def intrin_col2im(input_shape, output_shape, kernel, stride, pad, dtype):
    """
    Compute col2im via cce col2im intrin function call directly

    Args:
        input_shape: the shape of the image
        output_shape: the shape of the result of im2col given the input image
        kernel: kernel sizes for im2col
        stride: stride sizes for im2col
        pad: padding sizes for im2col, including padding top, bottom, left, and right
        dtype: type of the data

    Return:
        cce intrin function call for col2im
    """
    input_w, input_h = input_shape
    output_w, output_h = output_shape
    pad_left, pad_right, pad_top, pad_bottom = pad
    w_idx_kernel = 0
    h_idx_kernel = 0
    w_idx = (-pad_left) & 0xffff
    h_idx = (-pad_top) & 0xffff
    c1_idx = 0
    stride_w, stride_h = stride
    kernel_w, kernel_h = kernel
    dilation_w = dilation_h = 1

    jump_offset = 0
    repeat_mode = 0
    repeat_time = (output_w * output_h + 15) // 16
    input_b = 1
    input_c1 = 1
    input_h_tile = 1
    input_w_tile = 1
    input_c0 = 16

    input_shape = (input_b, input_c1, input_h_tile, input_w_tile, kernel_w, kernel_h, output_w, output_h, input_c0)
    input_data = akg.tvm.placeholder(input_shape, dtype=dtype)

    result = akg.tvm.compute((input_w, input_h, input_c0),
                             lambda h, w, c0:
                             input_data[0, 0, 0, 0,
                                        h // kernel_h,
                                        w // kernel_w,
                                        h % kernel_h, w % kernel_w,
                                        c0],
                             name='col2im_intrinsic')

    input_data_buff = akg.tvm.decl_buffer(input_data.shape, input_data.dtype,
                                          name="input_data_buff",
                                          offset_factor=1, scope="local.UB")

    result_buff = akg.tvm.decl_buffer(result.shape, result.dtype,
                                      name="result_buff",
                                      offset_factor=1, scope="local.UB")

    def pack_args(sp):
        if len(sp) != 20:
            raise RuntimeError("20 args are expected to pack but got {}"
                               "".format(len(sp)))
        # fcol2img = (sp[0] & 0xffff) << 0 | (sp[1] & 0xffff) << 16
        #            | (sp[2] & 0xff) << 32 | (sp[3] & 0xff) << 40
        #            | (sp[4] & 0xff) << 48 | (sp[5] & 0xff) << 56
        # Xm = (sp[6] & 0xff) << 16 | (sp[7] & 0xff) << 24
        #      | (sp[8] & 0xffff) << 32 | (sp[9] & 0xffff) << 48
        #      | (sp[10] & 0xfff) << 0
        # Xt = (sp[11] & 63) << 0 | (sp[12] & 63) << 6
        #      | (sp[13] & 0xff) << 12 | (sp[14] & 0xff) << 20
        #      | (sp[15] & 0xff) << 28 | (sp[16] & 0xff) << 36
        #      | (sp[17] & 0xff) << 44 | (sp[18] & 1) << 52 | (sp[19] & 0xff) << 56

        fcol2img = akg.tvm.const(sp[0], 'uint64') + akg.tvm.const(sp[1] * 2**16, 'uint64') \
            + akg.tvm.const(sp[2] * 2**32, 'uint64') + akg.tvm.const(sp[3] * 2**40, 'uint64') \
            + akg.tvm.const(sp[4] * 2**48, 'uint64') + akg.tvm.const(sp[5] * 2**56, 'uint64')
        xm = akg.tvm.const(sp[6] * 2**16, 'uint64') + akg.tvm.const(sp[7] * 2**24, 'uint64') \
            + akg.tvm.const(sp[8] * 2**32, 'uint64') + akg.tvm.const(sp[9] * 2**48, 'uint64') \
            + akg.tvm.const(sp[10], 'uint64')
        xt = akg.tvm.const(sp[11], 'uint64') + akg.tvm.const(sp[12] * 2**6, 'uint64') \
            + akg.tvm.const(sp[13] * 2**12, 'uint64') + akg.tvm.const(sp[14] * 2**20, 'uint64') \
            + akg.tvm.const(sp[15] * 2**28, 'uint64') + akg.tvm.const(sp[16] * 2**36, 'uint64') \
            + akg.tvm.const(sp[17] * 2**44, 'uint64') + akg.tvm.const(sp[18] * 2**52, 'uint64') \
            + akg.tvm.const(sp[19] * 2**56, 'uint64')

        return (fcol2img, xm, xt)

    def intrin_func(ins, outs):
        sp = [input_w, input_h, pad_left, pad_right, pad_top, pad_bottom,  # fmatrix
              w_idx_kernel, h_idx_kernel, w_idx, h_idx, c1_idx,  # xm
              stride_w, stride_h, kernel_w, kernel_h, dilation_w, dilation_h, jump_offset, repeat_mode, repeat_time]
        aa = ins[0]
        bb = outs[0]
        ib = akg.tvm.ir_builder.create()
        fcol2img, xm, xt = pack_args(sp)
        ib.emit(akg.tvm.call_extern(dtype, "set_fcol2img", fcol2img))
        ib.emit(akg.tvm.call_extern(dtype, "vector_dup",
                                    bb.access_ptr("w"), 0,
                                    (input_w * input_h * 16 + 63) // 64, 1, 1, 8, 8))
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                sp[6] = kw
                sp[7] = kh
                _, xm, xt = pack_args(sp)
                offset = (kh * kernel_h + kw) * output_h * output_w * 16
                ib.emit(akg.tvm.call_extern(dtype, "col2img", bb.access_ptr("rw"),
                                            aa.access_ptr("r", offset=offset), xm, xt))
        return ib.get()

    with akg.tvm.build_config(offset_factor=1):
        return akg.tvm.decl_tensor_intrin(result.op,
                                          intrin_func,
                                          binds={input_data: input_data_buff, result: result_buff})
