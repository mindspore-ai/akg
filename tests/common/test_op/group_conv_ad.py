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

"""operator dsl function: group_conv_ad"""

import akg.topi
import akg.tvm
import akg
import akg.lang.cce
import akg.backend as cce
from akg import dim
from test_op.conv_ad_v2 import expr_to_int


def set_dims_group(cut_h, cut_co, cut_m, cut_k, cut_n, out_shape_5d, _c_i, _c_o, group, _k_h, _k_w, _s_h, block_size):
    info = dim.Dim()
    out_n, out_c1, out_h, out_w, out_c0 = out_shape_5d
    tile_out_h = (cut_h - _k_h) // _s_h + 1
    if (out_n > 1):
        info.setdim(index=0, axis=0, tilel1=1, tilel0=0)
    if (out_c1 > 1):
        info.setdim(index=0, axis=0, tilel1=cut_co // block_size, tilel0=0)
    if (out_h > 1):
        info.setdim(index=0, axis='H', tilel1=tile_out_h, tilel0=0)
    if (out_w > 1):
        info.setdim(index=0, axis=3, tilel1=out_w, tilel0=0)
    if (out_c0 > 1):
        info.setdim(index=0, axis=4, tilel1=out_c0, tilel0=0)
    assert _c_i // block_size // group == 1
    if (_c_i // block_size // group > 1):
        info.setdim(index=0, axis=5, tilel1=_c_i // block_size // group, tilel0=0)
    if (_k_h > 1):
        info.setdim(index=0, axis=5, tilel1=_k_h, tilel0=0)
    if (_k_w > 1):
        info.setdim(index=0, axis=5, tilel1=_k_w, tilel0=0)
    return str(info)


@akg.tvm.register_func("akg.autodiff.group_conv_compute_forward")
def group_conv_forward(_n, _h, _w, _c_i, _c_o, group, _k_h, _k_w,
                       _a, _b, bias_value, pad_h, pad_w, _s_h, _s_w,
                       cut_h, cut_co, cut_m, cut_k, cut_n, block_size,
                       use_bias=False,
                       kernel_name='group_conv'):
    if (not isinstance(_n, int)):
        _n, _h, _w, _c_i, _c_o, group, _k_h, _k_w = expr_to_int((_n, _h, _w, _c_i, _c_o, group, _k_h, _k_w))
        pad_h, pad_w, _s_h, _s_w = expr_to_int((pad_h, pad_w, _s_h, _s_w))
        cut_h, cut_co, cut_m, cut_k, cut_n, block_size = expr_to_int((cut_h, cut_co, cut_m, cut_k, cut_n, block_size))

    conv_dtype = 'float16'

    if cut_h == _h:
        cut_h += pad_h + pad_h

    assert _c_o % group == 0 and _c_i % group == 0
    assert _c_o % block_size == 0 and (_c_i // group) % block_size == 0

    if (use_bias):
        bias = bias_value

    _o_h = (_h + 2 * pad_h - _k_h) // _s_h + 1
    _o_w = (_w + 2 * pad_w - _k_w) // _s_w + 1

    kc1 = akg.tvm.reduce_axis((0, _c_i // block_size // group), name='kc1')
    kh = akg.tvm.reduce_axis((0, _k_h), name='kh')
    kw = akg.tvm.reduce_axis((0, _k_w), name='kw')
    kc0 = akg.tvm.reduce_axis((0, block_size), name='kc0')

    p_top, p_bottom, p_left, p_right = pad_h, pad_h, pad_w, pad_w
    output_name = 'output'
    output_bias_name = 'output_bias'

    C = akg.tvm.compute((_n, _c_o // block_size, _o_h, _o_w, block_size),
                    lambda n, c1, h, w, c0:
                    akg.lang.cce.mmad(
                        akg.tvm.if_then_else(
                            akg.tvm.any((h * _s_h + kh) < p_top, (h * _s_h + kh) > (_h + p_top - 1),
                                        (w * _s_w + kw) < p_left, (w * _s_w + kw) > (_w + p_left - 1)),
                            akg.tvm.const(0.0, conv_dtype),
                            _a[n, c1 // ((_c_o // block_size) // group) * ((_c_i // block_size) // group) + kc1,
                               (h * _s_h + kh - p_top), (w * _s_w + kw - p_left), kc0])
                        * _b[(kc1 * _k_h + kh) * _k_w + kw, c1, c0, kc0],
                        axis=[kc1, kh, kw, kc0]),
        attrs={
                "pragma_conv_kernel_n": _c_o,
                "pragma_conv_kernel_h": _k_h,
                "pragma_conv_kernel_w": _k_w,
                "pragma_conv_padding_top": p_top,
                "pragma_conv_padding_bottom": p_bottom,
                "pragma_conv_padding_left": p_left,
                "pragma_conv_padding_right": p_right,
                "pragma_conv_bypass_l1": 1,
                "pragma_conv_stride_h": _s_h,
                "pragma_conv_stride_w": _s_w,
                "pragma_conv_fm_n": _n,
                "pragma_conv_fm_c": _c_i,
                "pragma_conv_fm_h": _h,
                "pragma_conv_fm_w": _w,
                "pragma_conv_dilation_h": 1,
                "pragma_conv_dilation_w": 1,
                "pragma_conv_h_cut": cut_h,
                "pragma_conv_w_cut": _w + 2 * pad_w,
                "pragma_conv_co_cut": cut_co,
                "pragma_conv_m_cut": cut_m,
                "pragma_conv_k_cut": cut_k,
                "pragma_conv_n_cut": cut_n,
                "feature": _a.op.name,
                "filter": _b.op.name,
                "bias": 'bias',
                "res": output_name,
                "res_bias": output_bias_name},
        name=output_name)

    if use_bias:
        out = akg.tvm.compute(C.shape,
                              lambda n, c1, h, w, c0:
                              C[n, c1, h, w, c0] + bias[0, c1, 0, 0, c0],
                              name=output_bias_name)
        bufs = [_a, _b, bias, out]
    else:
        out = C
        bufs = [_a, _b, out]

    # create schedule for cce
    s = akg.tvm.create_schedule([out.op])

    # set dim
    info = set_dims_group(cut_h, cut_co, cut_m, cut_k, cut_n,
                          expr_to_int(out.shape), _c_i, _c_o, group,
                          _k_h, _k_w, _s_h, block_size)

    # build
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=False):
        mod = akg.build(s, bufs, "cce", name=kernel_name, attrs={"dim": info}, polyhedral=True)

    return out

# _b in Fractal format; result in Fractal format
def group_flip_weight(_b, k_h, k_w, group, alpha, beta, block_size):
    hw = k_h * k_w

    out_shape = (alpha * hw, beta * group, block_size, block_size)
    b_group_flip = akg.tvm.compute(out_shape,
                                   lambda k0, k1, k2, k3: _b[(k1 % alpha) * hw + hw - 1 - k0 % hw,
                                                             (k1 // alpha) * beta + k0 // hw, k3, k2],
                                   name=_b.name + "_group_flipped")
    return b_group_flip

# _h in 5d format; result in 5d format
def strided_head(_h, s_h, s_w):
    n, c1, h, w, c0 = _h.shape
    out_shape = (n, c1, (h - 1) * s_h + 1, (w - 1) * s_w + 1, c0)
    h_strided = akg.tvm.compute(out_shape, lambda i0, i1, i2, i3, i4:
                            akg.tvm.expr.Select(akg.tvm.any(i2 % s_h != 0, i3 % s_w != 0),
                                            akg.tvm.const(0.0, dtype="float16"),
                                            _h[i0, i1, i2 // s_h, i3 // s_w, i4]), name=_h.name + "_strided")

    return h_strided

# _a in 5d format; result in 5d format
def transpose_data(_a, block_size):
    out_shape = (_a.shape[1].value * block_size, _a.shape[0].value // block_size,
                 _a.shape[2].value, _a.shape[3].value, block_size)

    a_transpose = akg.tvm.compute(out_shape,
                                  lambda j0, j1, j2, j3, j4:
                                  _a[j1 * block_size + j4, j0 // block_size, j2, j3, j0 % block_size],
                                  name=_a.name + "_transposed")
    return a_transpose

# _a in 5d format; result in 5d format
def transpose_regroup(_a, block_size, group):
    out_shape = ((_a.shape[1].value * block_size) // group, (_a.shape[0].value * group) // block_size,
                 _a.shape[2].value, _a.shape[3].value, block_size)
    _n = _a.shape[0].value
    CiG = (_a.shape[1].value * block_size) // group
    beta = CiG // block_size
    a_transpose_regroup = akg.tvm.compute(out_shape,
                                          lambda j0, j1, j2, j3, j4:
                                          _a[(j1 * block_size + j4) % _n, j0 // block_size
                                             + ((j1 * block_size + j4) // _n) * beta, j2, j3, j0 % block_size],
                                          name=_a.name + "_transposed_regrouped")
    return a_transpose_regroup


# head is in 5d format; result in Fractal format
def transpose_convert_head(head, block_size):
    out_shape = ((head.shape[0].value // block_size) * head.shape[2].value * head.shape[3].value,
                 head.shape[1].value, block_size, block_size)
    tmp_6d_shape = (head.shape[0].value // block_size, block_size,
                    head.shape[1].value, head.shape[2].value, head.shape[3].value, block_size)
    head_6d = akg.topi.reshape(head, tmp_6d_shape)
    head_6d_transpose = akg.topi.transpose(head_6d, (0, 3, 4, 2, 5, 1))
    head_transpose_convert = akg.topi.reshape(head_6d_transpose, out_shape)
    return head_transpose_convert


def group_conv_ad(_n, _h, _w, _c_i, _c_o, group, _k_h, _k_w, pad_h, pad_w, _s_h, _s_w,
                  cut_h, cut_co, cut_m, cut_k, cut_n, block_size, use_bias=False, kernel_name='group_conv'):
    conv_dtype = 'float16'
    _a = akg.tvm.placeholder((_n, _c_i // block_size, _h, _w, block_size), name="input0", dtype=conv_dtype)
    _b = akg.tvm.placeholder(((_c_i // group) // block_size * _k_h * _k_w, _c_o // block_size, block_size, block_size),
                             name="input1", dtype=conv_dtype)

    mod_forward = group_conv_forward(_n, _h, _w, _c_i, _c_o, group, _k_h, _k_w, _a, _b, None,
                                     pad_h, pad_w, _s_h, _s_w, cut_h, cut_co, cut_m, cut_k, cut_n, block_size)
    _o_h = mod_forward.shape[2].value
    _o_w = mod_forward.shape[3].value


    head = akg.tvm.placeholder(mod_forward.shape, name="head", dtype=conv_dtype)
    # (_n,_c_o,_o_h,_o_w)--(stride)-->(_n,_c_o,(_o_h-1)*_s_h+1,
    # (_o_w-1)*_s_w+1)--(5d)-->(_n,_c_o/16,(_o_h-1)*_s_h+1,(_o_w-1)*_s_w+1,16)
    pld_head_strided = akg.tvm.placeholder((_n, _c_o // block_size, (_o_h - 1) * _s_h + 1, (_o_w - 1) * _s_w + 1, block_size),
                                       name="head_strided_5d", dtype=conv_dtype)

    # (_c_o,_c_i//group,_k_h,_k_w)--(flip)-->
    # (_c_i,_c_o//group,_k_h,_k_w)--(Fractal)-->((_c_o//group)/16*_k_h*_k_w, _c_i/16,16,16)
    pld_b_flipped = akg.tvm.placeholder(((_c_o // group) // block_size * _k_h * _k_w, _c_i // block_size, block_size, block_size),
                                    name="b_flip", dtype=conv_dtype)

    # b in Fractal format; result in Fractal format
    b_group_flipped = group_flip_weight(_b, _k_h, _k_w, group, _c_o // group // block_size, _c_i // group // block_size, block_size)
    s_gr_fl = akg.tvm.create_schedule([b_group_flipped.op])
    info = dim.Dim()
    info.setdim(index=0, axis=0, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=1, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=2, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=3, tilel1=1, tilel0=1)

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=False):
        mod_b_group_flip = akg.build(s_gr_fl, [_b, b_group_flipped], "cce", name="b_group_flip",
                                    attrs={"dim": str(info)}, polyhedral=True)

    head_strided = strided_head(head, _s_h, _s_w)
    s_striding = akg.tvm.create_schedule(head_strided.op)

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=False):
        mod_head_strided = akg.build(s_striding, [head, head_strided], "cce", name="h_strided",
                                    attrs={"dim": str(info)}, polyhedral=True)


    a_transposed = transpose_regroup(_a, block_size, group)
    s_transposed_nc = akg.tvm.create_schedule(a_transposed.op)
    info = dim.Dim()
    info.setdim(index=0, axis=0, tilel1=16, tilel0=16)
    info.setdim(index=0, axis=1, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=2, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=3, tilel1=1, tilel0=1)

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_transposed_nc = akg.build(s_transposed_nc, [_a, a_transposed], "cce", name="a_transposed",
                                     attrs={"dim": str(info)}, polyhedral=True)

    head_transposed_convert = transpose_convert_head(head, block_size)
    s_transposed_convert = akg.tvm.create_schedule(head_transposed_convert.op)
    info = dim.Dim()
    info.setdim(index=0, axis=0, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=1, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=2, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=3, tilel1=1, tilel0=1)

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_transposed_convert = akg.build(s_transposed_convert, [head, head_transposed_convert], "cce",
                                           name="a_transposed", attrs={"dim": str(info)}, polyhedral=True)


    # Begin with the ad kernels
    ad_attrs = {"ad_conv_enable": 1}
    _jacs_data = list(akg.differentiate(mod_forward, [_a], head, ad_attrs, [pld_head_strided, pld_b_flipped, None]))

    cut_h_e, cut_co_e, cut_m_e, cut_k_e, cut_n_e = ((_o_h - 1) * _s_h + 1 + 2 * (_k_h - 1 - pad_h), 16, _h * _w, 48, 16)
    cut_m_e = ((cut_m_e + block_size - 1) // block_size) * block_size

    info = set_dims_group(cut_h_e, cut_co_e, cut_m_e, cut_k_e, cut_n_e,
                          expr_to_int(_a.shape), _c_o, _c_i, group, _k_h, _k_w, _s_h, block_size)

    s_data = akg.tvm.create_schedule([_jacs_data[0].op])
    # low_data = akg.lower(s_data, [pld_head_strided, pld_b_flipped, _jacs_data[0]], simple_mode=True)

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=False):
        mod_ad_data = akg.build(s_data, [pld_head_strided, pld_b_flipped, _jacs_data[0]], "cce",
                                name="conv_ad_data", attrs={"dim": info}, polyhedral=True)

    # (_n,_c_i,_h,_w)--(trans)-->(_c_i,_n,_h,_w)--(regroup)-->
    # (_c_i//group,_n*group,_h,_w)--(5d)-->(_c_i//group,(_n*group)/16,_h,_w,16)
    pld_x_trans = akg.tvm.placeholder((_c_i // group, (_n * group) // block_size, _h, _w, block_size),
                                      name="x_trans_5d", dtype=conv_dtype)

    # (_n,_c_o,_o_h,_o_w)--(trans)-->
    # (_c_o,_n,_o_h,_o_w)--(Fractal)-->(_n/16*_o_h*_o_w, _c_o/16,16,16)
    pld_head_trans_converted = akg.tvm.placeholder((_n // block_size * _o_h * _o_w, _c_o // block_size, block_size, block_size),
                                                   name="head_trans_convert", dtype=conv_dtype)

    # ad_attrs = {"ad_conv_enable": 1}
    _jacs_weights = list(akg.differentiate(mod_forward, [_b], head, ad_attrs,
                                           [pld_x_trans, pld_head_trans_converted, None]))

    cut_h_e, cut_co_e, cut_m_e, cut_k_e, cut_n_e = (_h + 2 * pad_h, 16, _k_h * _k_w, 48, 16)
    cut_m_e = ((cut_m_e + block_size - 1) // block_size) * block_size

    info = set_dims_group(cut_h_e, cut_co_e, cut_m_e, cut_k_e, cut_n_e,
                          (_c_i // group, _c_o // block_size, _k_h, _k_w, block_size),
                          _n * group, _c_o, group, _o_h, _o_w, 1, block_size)

    s_weights = akg.tvm.create_schedule([_jacs_weights[0].op])

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_ad_weights = akg.build(s_weights, [pld_x_trans, pld_head_trans_converted, _jacs_weights[0]], "cce",
                                   name="conv_ad_weights", attrs={"dim": info}, polyhedral=True)


    print("Forward input data shape: ", _a.shape)
    print("Forward input weight shape: ", _b.shape)
    print("Forward output shape: ", mod_forward.shape)
    print("Backward wrt. DATA input data shape: ", pld_head_strided.shape)
    print("Backward wrt. DATA input weight shape: ", pld_b_flipped.shape)
    print("Backward wrt. DATA output shape: ", _jacs_data[0].shape)
    print("Backward wrt. WEIGHT input data shape: ", pld_x_trans.shape)
    print("Backward wrt. WEIGHT input weight shape: ", pld_head_trans_converted.shape)
    print("Backward wrt. WEIGHT output shape: ", _jacs_weights[0].shape)

    return mod_ad_data, mod_ad_weights, mod_b_group_flip, mod_head_strided, mod_transposed_nc, mod_transposed_convert
