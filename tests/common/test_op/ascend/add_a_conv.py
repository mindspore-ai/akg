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

"""dsl: add_a_conv"""
import akg.tvm
import akg
from akg import dim
from akg.utils import kernel_exec as utils


def add_a_conv_compute(fmap_shape, filter_shape, pad_, stride_, dilation_,
                       tile_hh=0, tile_coco=0, tile_mm=0, tile_kk=0, tile_nn=0, bypass_l1=False,
                       use_bias=False, block_size=16, conv_dtype='float16'):
    # input shape (NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape
    in_c = (in_c + block_size - 1) // block_size * block_size
    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    k_c = (k_c + block_size - 1) // block_size * block_size
    k_n = (k_n + block_size - 1) // block_size * block_size
    # padding((padding_h, padding_w) -> (padding_top, padding_bottom, padding_left, padding_right))
    padding = (pad_[0], pad_[0], pad_[1], pad_[1])
    p_top, p_bottom, p_left, p_right = padding

    # stride (stride_h, stride_w)
    s_h, s_w = stride_

    # dilation (dilation_h, dilation_w)
    d_h, d_w = dilation_

    if tile_hh == in_h:
        tile_hh += p_top + p_bottom
    tile_coco = (tile_coco + block_size - 1) // block_size * block_size
    tile_mm = (tile_mm + block_size - 1) // block_size * block_size
    tile_kk = (tile_kk + block_size - 1) // block_size * block_size
    tile_nn = (tile_nn + block_size - 1) // block_size * block_size

    c0 = block_size
    c1_cut = tile_coco // c0
    h_window_cut = (tile_hh - k_h) // s_h + 1

    out_w = (in_w + p_left + p_right - k_w) // (s_w) + 1

    kernel_name = "add_a_conv_layer_" + str(in_n) + "_" + str(in_c) + "_" + str(in_h) + "_" + str(in_w) \
                  + "_" + str(k_n) + "_" + str(in_c) + "_" + str(k_h) + "_" + str(k_w) \
                  + "_" + str(p_top) + "_" + str(s_h)

    input_shape_nc1hwc0 = (in_n, in_c // block_size, in_h, in_w, block_size)
    in_n, in_c1, in_h, in_w, in_c0 = input_shape_nc1hwc0

    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0
    kernel_shape_fractal = (k_c // block_size * k_h * k_w, k_n // block_size, block_size, block_size)

    # bias shape
    bias_shape_nc1hwc0 = (1, k_n // block_size, 1, 1, block_size)

    # a_value placeholder (NC1HWCO)
    a_tmp = akg.tvm.placeholder(input_shape_nc1hwc0, dtype=conv_dtype, name='a_tmp')
    a_value = akg.tvm.compute(a_tmp.shape, lambda n, kc1, h, w, kc0: a_tmp[n, kc1, h, w, kc0] + 1, \
            name='a_value', attrs={'no_inline': 1})
    # b_value placeholder (fractal)
    b_value = akg.tvm.placeholder(kernel_shape_fractal, dtype=conv_dtype, name='b_value')

    if use_bias:
        bias_name = 'bias'
        bias_value = akg.tvm.placeholder(bias_shape_nc1hwc0, dtype=conv_dtype, name=bias_name)
    else:
        bias_name = 'None'
        bias_value = None

    # Create reduction variables
    kc1 = akg.tvm.reduce_axis((0, k_c1), name='kc1')
    kh = akg.tvm.reduce_axis((0, k_h), name='kh')
    kw = akg.tvm.reduce_axis((0, k_w), name='kw')
    kc0 = akg.tvm.reduce_axis((0, k_c0), name='kc0')

    k_h_d = (k_h - 1) * d_h + 1
    k_w_d = (k_w - 1) * d_w + 1
    out_h = (in_h + p_top + p_bottom - k_h_d) // (s_h) + 1
    tile_out_h = (tile_hh - k_h_d) // s_h + 1
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1

    out_shape_nc1hwc0 = (in_n, k_n // block_size, out_h, out_w, block_size)
    _, out_c1, out_h, out_w, out_c0 = out_shape_nc1hwc0

    if tile_coco > 0:
        c1_cut = tile_coco // block_size
    else:
        c1_cut = out_c1

    # set dim
    if s_h > k_h:
        a_cut_h = tile_out_h * s_h
    else:
        a_cut_h = (tile_out_h - 1) * s_h + k_h_d
    a_cut_w = (out_w - 1) * s_w + k_w_d

    index = 0
    info = dim.Dim()
    if in_c1 > 1:
        info.setdim(index=index, axis="C1", tilel1=in_c1, tilel0=in_c1)  # c1
    if in_h > 1:
        info.setdim(index=index, axis="H", tilel1=a_cut_h, tilel0=a_cut_h)  # h
    if in_w > 1:
        info.setdim(index=index, axis="W", tilel1=a_cut_w, tilel0=a_cut_w)  # w
    if in_c0 > 1:
        info.setdim(index=index, axis="C0", tilel1=in_c0, tilel0=in_c0)  # c0

    index += 1
    if out_c1 > 1:
        info.setdim(index=index, axis="C1", tilel1=c1_cut, tilel0=0)  # c1
    if out_h > 1:
        info.setdim(index=index, axis="H", tilel1=tile_out_h, tilel0=0)  # h
    if out_w > 1:
        info.setdim(index=index, axis="W", tilel1=out_w, tilel0=0)  # w
    if out_c0 > 1:
        info.setdim(index=index, axis="C0", tilel1=out_c0, tilel0=0)  # c0
    if in_c1 > 1:
        info.setdim(index=index, axis=5, tilel1=in_c1, tilel0=0)  # kc1
    if k_h > 1:
        info.setdim(index=index, axis=5, tilel1=k_h, tilel0=0)  # kh
    if k_w > 1:
        info.setdim(index=index, axis=5, tilel1=k_w, tilel0=0)  # kw

    # Compute the convolution
    output_name = "c_value"
    output_bias_name = "OUT"
    c_value = akg.tvm.compute(out_shape_nc1hwc0,
                    lambda n, c1, h, w, c0: akg.lang.ascend.mmad(
                        akg.tvm.if_then_else(akg.tvm.any((h * s_h + kh) < p_top, (h * s_h + kh) > (in_h + p_top - 1),
                                                 (w * s_w + kw) < p_left, (w * s_w + kw) > (in_w + p_left - 1)),
                                         akg.tvm.const(0.0, 'float16'),
                                         a_value[n, kc1, (h * s_h + (kh * d_h) - p_top), \
                                                 (w * s_w + (kw * d_w) - p_left), kc0])
                        * b_value[(kc1 * k_h + kh) * k_w + kw, c1, c0, kc0],
                        axis=[kc1, kh, kw, kc0]), name=output_name,
                    attrs={
                        "pragma_conv_kernel_n": k_n,
                        "pragma_conv_kernel_h": k_h,
                        "pragma_conv_kernel_w": k_w,
                        "pragma_conv_padding_top": p_top,
                        "pragma_conv_padding_bottom": p_bottom,
                        "pragma_conv_padding_left": p_left,
                        "pragma_conv_padding_right": p_right,
                        "pragma_conv_bypass_l1": 1 if bypass_l1 else 0,
                        "pragma_conv_stride_h": s_h,
                        "pragma_conv_stride_w": s_w,
                        "pragma_conv_dilation_h": d_h,
                        "pragma_conv_dilation_w": d_w,
                        "pragma_conv_fm_n": in_n,
                        "pragma_conv_fm_c": in_c,
                        "pragma_conv_fm_h": in_h,
                        "pragma_conv_fm_w": in_w,
                        "pragma_conv_h_cut": (h_window_cut - 1) * s_h + k_h_d,
                        "pragma_conv_w_cut": (in_w + p_left + p_right),
                        "pragma_conv_co_cut": c1_cut * k_c0,
                        "pragma_conv_m_cut": tile_mm,
                        "pragma_conv_k_cut": tile_kk,
                        "pragma_conv_n_cut": tile_nn,
                        "feature": a_value.op.name,
                        "filter": b_value.op.name,
                        "bias": bias_name,
                        "res": output_name,
                        "res_bias": output_bias_name})

    if use_bias:
        cube = akg.tvm.compute(out_shape_nc1hwc0, lambda n, c1, h, w, c0: c_value[n, c1, h, w, c0] + bias_value[0, c1, 0, 0, c0],
                           name=output_bias_name)
    else:
        cube = c_value
    return cube, a_tmp, b_value, bias_value, kernel_name, str(info)


def add_a_conv(fmap_shape, filter_shape, pad_, stride_, dilation_,
               tile_hh=0, tile_coco=0, tile_mm=0, tile_kk=0, tile_nn=0, bypass_l1=False,
               use_bias=False, block_size=16, conv_dtype='float16'):
    conv, a_value, b_value, bias_value, kernel_name, dim_info = add_a_conv_compute(fmap_shape, filter_shape, pad_, stride_, dilation_,
                                                                 tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, bypass_l1,
                                                                 use_bias, block_size, conv_dtype)
    # schedule
    s = akg.tvm.create_schedule(conv.op)
    print(conv, a_value, b_value, bias_value)

    attrs = {}
    attrs["pragma_rmselfdep"] = False
    attrs['dim'] = dim_info
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):

        if use_bias:
            mod = akg.build(s, [a_value, b_value, bias_value, conv], "cce", name=kernel_name, attrs=attrs, polyhedral=True)
        else:
            mod = akg.build(s, [a_value, b_value, conv], "cce", name=kernel_name, attrs=attrs, polyhedral=True)
    source_code = mod.imported_modules[0].get_source()
    cce_path = '.'
    utils.create_code(kernel_name, cce_path, source_code)

    return mod


def conv_relu(fmap_shape, filter_shape, pad_, stride_, dilation_,
              tile_hh=0, tile_coco=0, tile_mm=0, tile_kk=0, tile_nn=0, bypass_l1=False,
              use_bias=False, block_size=16, conv_dtype='float16'):
    conv, a_value, b_value, bias_value, kernel_name, dim_info = add_a_conv_compute(fmap_shape, filter_shape, pad_, stride_, dilation_,
                                                                 tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, bypass_l1,
                                                                 use_bias, block_size, conv_dtype)
    # leakly relu
    negative_slope = 0.0
    slope_tmp = akg.tvm.const(negative_slope, dtype=conv_dtype)
    # negative_slope*x
    out = akg.lang.ascend.vmuls(conv, slope_tmp)
    # max(x,negative_slope*x)
    out = akg.lang.ascend.vmax(out, conv)
    # schedule
    s = akg.tvm.create_schedule(conv.op)
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):

        if use_bias:
            mod = akg.build(s, [a_value, b_value, bias_value, conv], "cce", name=kernel_name, attrs={"dim": dim_info}, polyhedral=True)
        else:
            mod = akg.build(s, [a_value, b_value, conv], "cce", name=kernel_name, attrs={"dim": dim_info}, polyhedral=True)
    return mod
