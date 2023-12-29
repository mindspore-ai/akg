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

"""operator dsl function: conv_backprop_filter"""
import akg.tvm
import akg
from akg import dim
import akg.utils as utils
from akg.utils.validation_check import comp_conv_backprop_out_shape


batch_conv_backprop_filter_tiling_args = {
    str(((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [16, 1, 1, 288, 1, 13, 13, 288, 80, 16],
    str(((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [32, 1, 1, 256, 1, 14, 14, 256, 96, 32],
    str(((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [32, 1, 1, 512, 1, 10, 14, 512, 16, 32],
    str(((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [32, 1, 1, 416, 1, 13, 13, 416, 64, 32],
    str(((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
        [16, 3, 3, 32, 1, 30, 30, 32, 64, 144],
    str(((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [64, 1, 1, 32, 1, 28, 28, 32, 304, 64],
    str(((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))):
        [16, 3, 3, 32, 1, 19, 57, 32, 64, 144],
    str(((32, 16, 224, 224), (64, 16, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1))):
        [16, 7, 7, 16, 1, 25, 229, 16, 16, 784],
    str(((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [64, 1, 1, 416, 1, 7, 7, 416, 32, 64],
    str(((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [256, 1, 1, 32, 1, 14, 14, 32, 32, 256],
    str(((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
        [16, 3, 3, 128, 1, 16, 16, 128, 48, 144],
    str(((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))):
        [16, 3, 3, 64, 1, 21, 29, 64, 176, 144],
    str(((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [32, 1, 1, 32, 1, 20, 56, 32, 256, 32],
    str(((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [16, 1, 1, 64, 1, 55, 55, 64, 96, 16],
    str(((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [32, 1, 1, 64, 1, 55, 55, 64, 80, 32],
    str(((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [32, 1, 1, 16, 1, 56, 56, 16, 1008, 32],
    str(((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1))):
        [16, 1, 7, 16, 1, 23, 229, 16, 272, 112],
    str(((32, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))):
        [16, 7, 7, 16, 1, 56, 224, 448, 32, 64],
    str(((32, 3, 227, 227), (96, 3, 11, 11), (0, 0, 0, 0), (4, 4), (1, 1))):
        [16, 11, 11, 16, 1, 11, 227, 16, 16, 1936],
    str(((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))):
        [16, 3, 3, 256, 1, 15, 15, 256, 128, 144],
    str(((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [16, 1, 1, 1024, 1, 19, 27, 1024, 16, 16],
    str(((32, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [64, 1, 1, 32, 1, 22, 28, 32, 240, 64],
    str(((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [128, 1, 1, 32, 1, 28, 28, 32, 256, 128],
    str(((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [16, 1, 1, 256, 1, 19, 27, 256, 96, 16],
    str(((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [256, 1, 1, 16, 1, 7, 7, 16, 16, 256],
    str(((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
        [16, 3, 3, 96, 1, 9, 7, 96, 16, 144],
    str(((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [16, 1, 1, 32, 1, 31, 56, 32, 928, 16],
    str(((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [16, 1, 1, 16, 1, 51, 56, 16, 896, 16],
    str(((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
        [16, 3, 3, 16, 1, 58, 56, 16, 112, 144],
}


def gen_key(fmap_shape, filter_shape, pad_, stride_, dilation_):
    """generate key."""
    batch_size = fmap_shape[0]
    key_fmap_shape_start_index = 0
    if batch_size == 1:
        key_fmap_shape_start_index = 1
    key = str((tuple(fmap_shape[key_fmap_shape_start_index:4]), tuple(filter_shape), tuple(pad_),
               tuple(stride_), tuple(dilation_)))
    return key


def conv_backprop_filter_compute(data, input_shape, filter_shape, output_shape, pad_, stride_, dilation_,
                                 block_size=16, attrs=None, key=None):
    """core computation of conv_backprop_filter_compute."""
    conv_backprop_filter_tiling_args = {
        str(((6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1))):
        [16, 5, 5, 16, 1, 14, 14, 65536, 16, 65536],
        str(((1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [64, 1, 1, 64, 1, 14, 14, 16, 64, 16],
        str(((1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [64, 1, 1, 64, 1, 14, 14, 16, 16, 16],
        str(((1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [128, 1, 1, 128, 1, 14, 14, 49, 32, 512],
        str(((128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
        [128, 3, 3, 128, 1, 28, 28, 32, 112, 128],
        str(((128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [64, 1, 1, 256, 1, 28, 28, 128, 112, 32],
        str(((2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [7, 1, 1, 512, 1, 7, 7, 49, 32, 512],
        str(((256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [128, 1, 1, 256, 1, 14, 14, 128, 16, 128],
        str(((256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
        [128, 3, 3, 128, 1, 14, 14, 128, 16, 128],
        str(((256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [128, 1, 1, 128, 1, 56, 56, 128, 112, 128],
        str(((256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [16, 1, 1, 64, 1, 56, 56, 280, 16, 64],
        str(((3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))):
        [16, 7, 7, 16, 1, 117, 224, 65536, 32, 65536],
        str(((16, 224, 224), (64, 16, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))):
        [16, 7, 7, 16, 1, 117, 224, 65536, 32, 65536],
        str(((512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [14, 1, 1, 128, 1, 28, 28, 448, 16, 64],
        str(((512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [128, 1, 1, 256, 1, 28, 28, 128, 16, 128],
        str(((512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [7, 1, 1, 128, 1, 7, 7, 49, 256, 128],
        str(((512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
        [256, 3, 3, 256, 1, 7, 7, 128, 16, 64],
        str(((64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [128, 1, 1, 64, 1, 56, 56, 64, 16, 128],
        str(((64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))):
        [64, 1, 1, 64, 1, 56, 56, 64, 16, 64],
        str(((64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
        [64, 3, 3, 64, 1, 56, 56, 64, 16, 64],
        str(((256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [64, 1, 1, 128, 1, 56, 56, 128, 16, 128],
        str(((512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))):
        [256, 1, 1, 512, 1, 28, 28, 256, 16, 128],
    }
    stride_h, stride_w = stride_
    if stride_h != stride_w:
        raise ValueError("stride_h must be equal to stride_w.")
    # conv_backprop_filter input shape (NCHW -> NC1HWC0 -> fractal): load2d L0A
    input_n, input_c, input_h, input_w = output_shape
    if input_c % block_size != 0:
        raise ValueError("output channel must be divided by block_size.")
    if input_n > 32:
        raise ValueError("Batch must be less than or equal to 32.")
    input_shape_nc1hwc0 = (input_n, input_c // block_size, input_h, input_w, block_size)
    input_n, input_c1, input_h, input_w, input_c0 = input_shape_nc1hwc0
    mo = (input_h * input_w + block_size - 1) // block_size
    mi = block_size
    input_trans_shape_fractal = (input_n, input_c1, mo, input_c0, mi)

    # conv_backprop_filter kernel shape (NCHW -> NC1HWC0): img2col L0B
    k_n, k_c, k_h, k_w = input_shape
    if k_c % block_size != 0:
        raise ValueError("input channel must be divided by block_size.")
    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0

    # conv_backprop_filter output shape (NCHW -> NC1HWC0)
    out_n, out_c, out_h, out_w = filter_shape
    if out_n != input_c:
        raise ValueError("out_n must be equal to input_c.")
    output_shape_nc1hwc0 = (out_n, out_c // block_size, out_h, out_w, block_size)
    out_n, out_c1, out_h, out_w, _ = output_shape_nc1hwc0
    output_shape_fractal = (out_c1, out_h, out_w, out_n // block_size, block_size, block_size)
    out_c1, out_h, out_w, out_mo, out_mi, out_ni = output_shape_fractal

    padding = (pad_[0], pad_[1], pad_[2], pad_[3])
    p_top, p_bottom, p_left, p_right = padding

    s_h, s_w = stride_

    data_a = data[0]
    o_n, o_c1, o_h, o_w, o_c0 = data_a.shape
    mo = (o_h * o_w + block_size - 1) // block_size
    mi = block_size
    a_shape_fractal = (o_n, o_c1, mo, mi, o_c0)
    a_fractal = akg.tvm.placeholder(a_shape_fractal, dtype=data_a.dtype, name="backprop")
    a_buf = akg.tvm.decl_buffer(a_shape_fractal, a_fractal.dtype, name="backprop")
    data_b = data[1]
    tiling_args = batch_conv_backprop_filter_tiling_args
    use_autotiling = False
    if k_n == 1:
        tiling_args = conv_backprop_filter_tiling_args
    if attrs is not None and 'conv_tile' in attrs and len(attrs['conv_tile']) >= 8:
        tile = attrs['conv_tile']
    elif key in tiling_args:
        tile = tiling_args[key]
    else:
        use_autotiling = True

    in_h = k_h
    in_w = k_w
    if not use_autotiling:
        # set dim
        info = dim.Dim()
        index_ = 0

        tile_ci = tile[0]
        if tile_ci > k_c1 * k_c0:
            tile_ci = k_c1 * k_c0
        tile_ci = (tile_ci + block_size - 1) // block_size

        tile_kh = tile[1]
        if tile_kh > out_h:
            tile_kh = out_h

        tile_kw = tile[2]
        if tile_kw > out_w:
            tile_kw = out_w

        tile_coco = tile[3]
        if tile_coco > input_c1 * input_c0:
            tile_coco = input_c1 * input_c0
        tile_coco = (tile_coco + block_size - 1) // block_size

        tile_batch = tile[4]
        if tile_batch > input_n:
            tile_batch = input_n
        if tile_batch != 1:
            raise ValueError("tile_batch must be 1.")

        d_h, d_w = dilation_

        tile_hh = tile[5]
        if tile_hh == in_h:
            tile_hh = in_h + p_top + p_bottom
        elif tile_hh > in_h + p_top + p_bottom:
            tile_hh = in_h + p_top + p_bottom
        h_win_cut = (tile_hh - ((out_h - 1) * d_h + 1)) // s_h + 1

        tile_ww = tile[6]
        if tile_ww == in_w:
            tile_ww = in_w + p_left + p_right
        elif tile_ww > in_w + p_left + p_right:
            tile_ww = in_w + p_left + p_right
        w_win_cut = (tile_ww - ((out_w - 1) * d_w + 1)) // s_w + 1

        tile_mm = tile[7]
        tile_kk = tile[8]
        tile_nn = tile[9]

        tile_mm = (tile_mm + block_size - 1) // block_size * block_size
        tile_kk = (tile_kk + block_size - 1) // block_size * block_size
        tile_nn = (tile_nn + block_size - 1) // block_size * block_size

        if out_c1 > 1:
            info.setdim(index=index_, axis=0, tilel1=tile_ci, tilel0=tile_ci)
        if out_h > 1:
            info.setdim(index=index_, axis=0, tilel1=tile_kh, tilel0=tile_kh)
        if out_w > 1:
            info.setdim(index=index_, axis=0, tilel1=tile_kw, tilel0=tile_kw)
        if out_mo > 1:
            info.setdim(index=index_, axis=0, tilel1=tile_coco, tilel0=tile_coco)
        if out_mi > 1:
            info.setdim(index=index_, axis=0, tilel1=out_mi, tilel0=out_mi)  # mi don't tile
        if out_ni > 1:
            info.setdim(index=index_, axis=0, tilel1=out_ni, tilel0=out_ni)  # ni don't tile
        if input_n > 1:
            info.setdim(index=index_, axis=0, tilel1=tile_batch, tilel0=tile_batch)  # Batch tile
        if k_h > 1:
            info.setdim(index=index_, axis="H", tilel1=h_win_cut, tilel0=h_win_cut)  # out_h
        if k_w > 1:
            info.setdim(index=index_, axis="W", tilel1=w_win_cut, tilel0=w_win_cut)  # out_w

        info = str(info)
    else:
        info = ""

    # Compute the convolution
    output_name = "filter"

    a_trans = akg.tvm.compute(input_trans_shape_fractal,
                              lambda n, co1, mo, co0, mi: a_fractal[n, co1, mo, mi, co0], name='dy_trans')

    # Create reduction variables
    no = akg.tvm.reduce_axis((0, input_n), name='no')
    ho = akg.tvm.reduce_axis((0, input_h), name='ho')
    wo = akg.tvm.reduce_axis((0, input_w), name='wo')

    conv_filter_attr = {
        "pragma_conv_kernel_n": out_n,
        "pragma_conv_kernel_h": out_h,
        "pragma_conv_kernel_w": out_w,
        "pragma_conv_padding_top": p_top,
        "pragma_conv_padding_bottom": p_bottom,
        "pragma_conv_padding_left": p_left,
        "pragma_conv_padding_right": p_right,
        "pragma_conv_bypass_l1": 0,
        "pragma_conv_backprop_filter": 1,
        "pragma_conv_stride_h": s_h,
        "pragma_conv_stride_w": s_w,
        "pragma_conv_dilation_h": 1,
        "pragma_conv_dilation_w": 1,
        "pragma_conv_fm_n": k_n,
        "pragma_conv_fm_c": k_c,
        "pragma_conv_fm_h": k_h,
        "pragma_conv_fm_w": k_w,
        "feature": data_b.op.name,
        "filter": a_fractal.op.name,
        "bias": 'None',
        "res": output_name}

    if not use_autotiling:
        conv_filter_attr["pragma_conv_batch_cut"] = tile_batch
        conv_filter_attr["pragma_conv_h_cut"] = (h_win_cut - 1) * s_h + ((out_h - 1) * d_h + 1)
        conv_filter_attr["pragma_conv_w_cut"] = (w_win_cut - 1) * s_w + ((out_w - 1) * d_w + 1)
        conv_filter_attr["pragma_conv_co_cut"] = tile_coco * block_size
        conv_filter_attr["pragma_conv_cin_cut"] = tile_ci * block_size
        conv_filter_attr["pragma_conv_m_cut"] = tile_mm
        conv_filter_attr["pragma_conv_k_cut"] = tile_kk
        conv_filter_attr["pragma_conv_n_cut"] = tile_nn
        conv_filter_attr["pragma_conv_kh_cut"] = tile_kh
        conv_filter_attr["pragma_conv_kw_cut"] = tile_kw

    res_c = akg.tvm.compute(output_shape_fractal,
                            lambda c1, h, w, mo, mi, ni: akg.lang.ascend.mmad(
                                (akg.tvm.if_then_else(akg.tvm.any((h + s_h * ho) < p_top,
                                                                  (h + s_h * ho) > (in_h + p_top - 1),
                                                                  (w + s_w * wo) < p_left,
                                                                  (w + s_w * wo) > (in_w + p_left - 1)),
                                                      akg.tvm.const(0.0, 'float16'),
                                                      a_trans[no, mo, (input_w * ho + wo) // 16,
                                                              mi, (input_w * ho + wo) % 16])
                                 * data_b[no, c1, (ho * s_h + h - p_top),
                                          (wo * s_w + w - p_left), ni]).astype("float32"),
                                axis=[no, ho, wo]), name=output_name, attrs=conv_filter_attr)

    return res_c, {"dim": info, "pragma_conv_special_dma": 1,
                   utils.BINDS: {data_a: a_buf, a_fractal: a_buf}}


@utils.check_input_type((list, tuple), (list, tuple), (list, tuple), (list, tuple), (list, tuple), (list, tuple),
                        (dict, type(None)), (str, type(None)))
def conv_backprop_filter(data, fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=None):
    """
    Computes dw according "conv forward".

    Args:
        data (list[tvm.tensor.Tensor]): list with length 2.
              data[0](consider as dy) Tensor of type float16 ,shape 5D(out_n, out_c//C0, out_h, out_w,C0)
              data[1](consider as x)  Tensor of type float16 ,shape 5D(fN,fC//C0,fH,fW,C0)
        fmap_shape (list[int]): [fN, fC, fH, fW]
        filter_shape (list[int]): [wN, wC, wH, wW]
        pad_ (list[int]): [pad_left, pad_right, pad_top, pad_bottom]
        stride_ (list[int]): [stride_h, stride_w]
        dilation_ (list[int]): [dilation_h, dilation_w]
        attrs (dict): a dict with keys like conv_tile,bypass.

    Returns:
        tvm.tensor.Tensor.
        configs.

    Supported Platforms:
        'Ascend'
    """

    if len(data) != 2:
        raise IndexError("data contains output tensor and feature map tensor")

    utils.convolution_format_check(fmap_shape, filter_shape, pad_, stride_, dilation_)

    block_size = 16
    dy_shape, dx_shape, dw_shape = comp_conv_backprop_out_shape(fmap_shape, filter_shape, pad_, stride_, dilation_)

    key = gen_key(fmap_shape, filter_shape, pad_, stride_, dilation_)
    res_c, configs = conv_backprop_filter_compute(data, dx_shape, dw_shape, dy_shape, pad_, stride_, dilation_,
                                                  block_size=block_size, attrs=attrs, key=key)

    return res_c, configs
