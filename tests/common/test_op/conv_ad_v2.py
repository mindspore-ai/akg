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

"""operator dsl function: conv_ad_v2"""
import akg.topi
import akg.tvm
import akg
import akg.lang.cce
import akg.backend as cce
from akg import dim
from akg.ops.nn import conv as conv_origin
from akg.tvm import truncdiv, truncmod, floordiv

def set_dims(fmap_shape, filter_shape, pad_, stride_, dilation_, tile_hh,
             tile_coco, tile_mm, tile_kk, tile_nn, block_size):
    """set dim info in attrs."""
    in_n, in_c, in_h, in_w = fmap_shape
    in_c = (in_c + block_size - 1) // block_size * block_size
    in_c1 = in_c // block_size

    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    k_c = (k_c + block_size - 1) // block_size * block_size
    k_n = (k_n + block_size - 1) // block_size * block_size

    padding = (pad_[0], pad_[0], pad_[1], pad_[1])
    p_top, p_bottom, p_left, p_right = padding
    s_h, s_w = (stride_[0], stride_[1])
    d_h, d_w = (dilation_[0], dilation_[1])
    if (tile_hh == in_h):
        tile_hh += p_top + p_bottom
    tile_coco = (tile_coco + block_size - 1) // block_size * block_size
    tile_mm = (tile_mm + block_size - 1) // block_size * block_size
    tile_kk = (tile_kk + block_size - 1) // block_size * block_size
    tile_nn = (tile_nn + block_size - 1) // block_size * block_size


    k_h_d = (k_h - 1) * d_h + 1
    k_w_d = (k_w - 1) * d_w + 1
    out_h = (in_h + p_top + p_bottom - k_h_d) // (s_h) + 1
    tile_out_h = (tile_hh - k_h_d) // s_h + 1
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1

    out_shape_nc1hwc0 = (in_n, k_n // block_size, out_h, out_w, block_size)
    out_n, out_c1, out_h, out_w, out_c0 = out_shape_nc1hwc0

    if (tile_coco > 0):
        c1_cut = tile_coco // block_size
    else:
        c1_cut = out_c1

    # set dim
    info = dim.Dim()
    if (out_n > 1):
        info.setdim(index=0, axis=0, tilel1=1, tilel0=0)  # n
    if (out_c1 > 1):
        info.setdim(index=0, axis=0, tilel1=c1_cut, tilel0=0)  # c1
    if (out_h > 1):
        info.setdim(index=0, axis="H", tilel1=tile_out_h, tilel0=0)  # h
    if (out_w > 1):
        info.setdim(index=0, axis=3, tilel1=out_w, tilel0=0)  # w
    if (out_c0 > 1):
        info.setdim(index=0, axis=4, tilel1=out_c0, tilel0=0)  # c0

    if (in_c1 > 1):
        info.setdim(index=0, axis=5, tilel1=in_c1, tilel0=0)  # kc1
    if (k_h > 1):
        info.setdim(index=0, axis=5, tilel1=k_h, tilel0=0)  # kh
    if (k_w > 1):
        info.setdim(index=0, axis=5, tilel1=k_w, tilel0=0)  # kw

    return str(info)


def expr_to_int(A):
    result = []
    for i in range(len(A)):
        result.append(A[i].value)
    return result


@akg.tvm.register_func("akg.autodiff.conv_compute_forward")
def conv_compute_forward(fmap_shape, filter_shape, pad_, stride_, dilation_, A, B, bias_value=None,
                         tile_hh=0, tile_coco=0, tile_mm=0, tile_kk=0, tile_nn=0, bypass_l1=False,
                         use_bias=False, block_size=16, conv_dtype='float16'):
    if (not isinstance(fmap_shape[0], int)):
        fmap_shape = expr_to_int(fmap_shape)
    if (not isinstance(filter_shape[0], int)):
        filter_shape = expr_to_int(filter_shape)
    if (not isinstance(pad_[0], int)):
        pad_ = expr_to_int(pad_)
    if (not isinstance(stride_[0], int)):
        stride_ = expr_to_int(stride_)
    if (not isinstance(dilation_[0], int)):
        dilation_ = expr_to_int(dilation_)

    # input shape (NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape

    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape

    # padding((padding_h, padding_w) -> (padding_top, padding_bottom, padding_left, padding_right))
    padding = (pad_[0], pad_[0], pad_[1], pad_[1])
    p_top, p_bottom, p_left, p_right = padding

    # stride (stride_h, stride_w)
    s_h, s_w = stride_

    # dilation (dilation_h, dilation_w)
    d_h, d_w = dilation_

    if (tile_hh == in_h):
        tile_hh += p_top + p_bottom
    tile_coco = (tile_coco + block_size - 1) // block_size * block_size
    tile_mm = (tile_mm + block_size - 1) // block_size * block_size
    tile_kk = (tile_kk + block_size - 1) // block_size * block_size
    tile_nn = (tile_nn + block_size - 1) // block_size * block_size

    h_window_cut = (tile_hh - k_h) // s_h + 1

    input_shape_nc1hwc0 = (in_n, in_c // block_size, in_h, in_w, block_size)
    in_n, _, in_h, in_w, _ = input_shape_nc1hwc0

    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0


    # bias shape
    bias_shape_nc1hwc0 = (1, k_n // block_size, 1, 1, block_size)

    if use_bias:
        bias_name = 'input2'
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
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1

    out_shape_nc1hwc0 = (in_n, k_n // block_size, out_h, out_w, block_size)
    _, out_c1, out_h, out_w, _ = out_shape_nc1hwc0

    if (tile_coco > 0):
        c1_cut = tile_coco // block_size
    else:
        c1_cut = out_c1

    # set dim
    info = set_dims(fmap_shape, filter_shape, pad_, stride_, dilation_,
                    tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)

    # Compute the convolution
    output_name = "output0"
    output_bias_name = "output1"
    C = akg.tvm.compute(out_shape_nc1hwc0,
                    lambda n, c1, h, w, c0: akg.lang.cce.mmad(
                        akg.tvm.if_then_else(akg.tvm.any((h * s_h + kh) < p_top, (h * s_h + kh) > (in_h + p_top - 1),
                                                 (w * s_w + kw) < p_left, (w * s_w + kw) > (in_w + p_left - 1)),
                                         akg.tvm.const(0.0, 'float16'),
                                         A[n, kc1, (h * s_h + (kh * d_h) - p_top), (w * s_w + (kw * d_w) - p_left), kc0])
                        * B[(kc1 * k_h + kh) * k_w + kw, c1, c0, kc0],
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
                        "feature": A.op.name,
                        "filter": B.op.name,
                        "bias": bias_name,
                        "res": output_name,
                        "res_bias": output_bias_name})

    if use_bias:
        cube = akg.tvm.compute(out_shape_nc1hwc0,
                               lambda n, c1, h, w, c0: C[n, c1, h, w, c0] + bias_value[0, c1, 0, 0, c0],
                               name=output_bias_name)
    else:
        cube = C

    return cube


def conv_01(fmap_shape, filter_shape, pad_, stride_, dilation_,
            tile_hh=0, tile_coco=0, tile_mm=0, tile_kk=0, tile_nn=0,
            use_bias=False, block_size=16, conv_dtype='float16'):

    # input shape (NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape
    in_c = (in_c + block_size - 1) // block_size * block_size
    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    k_c = (k_c + block_size - 1) // block_size * block_size
    k_n = (k_n + block_size - 1) // block_size * block_size

    input_shape_nc1hwc0 = (in_n, in_c // block_size, in_h, in_w, block_size)

    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, _, k_h, k_w, _ = kernel_shape_nc1hwc0
    kernel_shape_fractal = (k_c // block_size * k_h * k_w, k_n // block_size, block_size, block_size)


    # A placeholder (NC1HWCO)
    A = akg.tvm.placeholder(input_shape_nc1hwc0, dtype=conv_dtype, name="input0")
    # B_placeholder (fractal)
    B = akg.tvm.placeholder(kernel_shape_fractal, dtype=conv_dtype, name="input1")
    data = [A, B]
    if use_bias:
        bias_shape_nc1hwc0 = (1, k_n // block_size, 1, 1, block_size)
        bias_name = "input2"
        bias_value = akg.tvm.placeholder(bias_shape_nc1hwc0, dtype=conv_dtype, name=bias_name)
        data.append(bias_value)
    else:
        bias_name = 'None'
        bias_value = None

    conv, _ = conv_origin.conv(data, fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias)

    kernel_name = 'conv_ad'

    k_n, k_c, k_h, k_w = filter_shape
    k_c = (k_c + block_size - 1) // block_size * block_size
    k_n = (k_n + block_size - 1) // block_size * block_size
    k_hw = k_h * k_w
    const_shift = k_hw - 1

    # B in Fractal format; result in Fractal format
    def flip_weight(B, k_c, k_hw, const_shift):
        out_shape = (B.shape[1].value * k_hw, k_c // block_size, block_size, block_size)
        B_flip = akg.tvm.compute(out_shape,
                                 lambda i0, i1, i2, i3: B[i1 * k_hw + const_shift - truncmod(i0, k_hw),
                                                          floordiv(i0, k_hw), i3, i2],
                                 name=B.name + "_flipped")
        return B_flip

    def strided_head(H, s_h, s_w):
        n, c1, h, w, c0 = H.shape
        out_shape = (n, c1, (h - 1) * s_h + 1, (w - 1) * s_w + 1, c0)
        H_strided = akg.tvm.compute(out_shape, lambda i0, i1, i2, i3, i4:
                                    akg.tvm.expr.Select(akg.tvm.any(truncmod(i2, s_h) != 0,
                                                                    truncmod(i3, s_w) != 0),
                                                        akg.tvm.const(0.0, dtype="float16"),
                                                        H[i0, i1, floordiv(i2, s_h), floordiv(i3, s_w), i4]),
                                    name=H.name + "_strided")
        return H_strided

    B_flip = flip_weight(B, k_c, k_hw, const_shift)

    pld_B_flip = akg.tvm.placeholder(B_flip.shape, name="inp1_flipped", dtype='float16')
    HEAD = akg.tvm.placeholder(conv.shape, name="Head", dtype='float16')

    HEAD_n, HEAD_c1, HEAD_h, HEAD_w, HEAD_c0 = HEAD.shape
    info = set_dims((HEAD_n.value, HEAD_c1.value * HEAD_c0.value, HEAD_h.value, HEAD_w.value),
                    (k_c, k_n, k_h, k_w), (2, 2), (1, 1), (1, 1),
                    tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)

    s_h, s_w = stride_
    if (s_h == 1) and (s_w == 1):
        ad_attrs = {"ad_conv_enable": 1, "ad_conv_reuse_conv": 1}
        jacs = list(akg.differentiate(conv, [A], HEAD, ad_attrs, [HEAD, pld_B_flip, None]))
        sjac = akg.tvm.create_schedule([jacs[0].op])
        op_vars = [HEAD, pld_B_flip, jacs[0]]
        info = set_dims((HEAD_n.value, HEAD_c1.value * HEAD_c0.value, HEAD_h.value, HEAD_w.value),
                        (k_c, k_n, k_h, k_w), (k_h - 1, k_w - 1), (1, 1), (1, 1),
                        tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)
    else:
        Head_strided = strided_head(HEAD, s_h, s_w)
        pld_Head_strided = akg.tvm.placeholder(Head_strided.shape, name="head_strided", dtype='float16')

        ad_attrs = {"ad_conv_enable": 1, "ad_conv_reuse_conv": 1}
        jacs = list(akg.differentiate(conv, [A], HEAD, ad_attrs, [pld_Head_strided, pld_B_flip, None]))
        sjac = akg.tvm.create_schedule([jacs[0].op])
        op_vars = [pld_Head_strided, pld_B_flip, jacs[0]]
        h_n, h_c1, h_h, h_w, h_c0 = pld_Head_strided.shape
        info = set_dims((h_n.value, h_c1.value * h_c0.value, h_h.value, h_w.value), (k_c, k_n, k_h, k_w),
                        (k_h - 1, k_w - 1), (1, 1), (1, 1), tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)


    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_backward = akg.build(sjac, op_vars, "cce", name=kernel_name, attrs={"dim": str(info)}, polyhedral=True)


    def transpose_data(A):
        out_shape = (A.shape[1] * block_size, truncdiv(A.shape[0], block_size), A.shape[2], A.shape[3], block_size)
        A_transpose = akg.tvm.compute(out_shape,
                                      lambda j0, j1, j2, j3, j4:
                                      A[j1 * block_size + j4, truncdiv(j0, block_size), j2, j3, truncmod(j0, block_size)],
                                      name=A.name + "_transposed")
        return A_transpose

    # Head is in 5D format
    # Output is in Fractal format
    def transpose_convert_head(Head):
        out_shape = ((floordiv(Head.shape[0].value, block_size)) * Head.shape[2].value * Head.shape[3].value,
                     Head.shape[1].value, block_size, block_size)
        tmp_6D_shape = (floordiv(Head.shape[0].value, block_size),
                        block_size, Head.shape[1].value, Head.shape[2].value, Head.shape[3].value, block_size)

        Head_6D = akg.topi.reshape(Head, tmp_6D_shape)
        # Transpose from (N//block_size_N, block_size_N, C//block_size_C, H, W, block_size_C)
        #           to   (N//block_size_N, H, W, C//block_size_C, block_size_C, block_size_N,)
        Head_6D_transpose = akg.topi.transpose(Head_6D, (0, 3, 4, 2, 5, 1))
        Head_transpose_convert = akg.topi.reshape(Head_6D_transpose, out_shape)
        return Head_transpose_convert


    X_transposed = transpose_data(A)
    pld_X_transposed = akg.tvm.placeholder(X_transposed.shape, name="inp0_transposed", dtype='float16')

    if (s_h > 1) or (s_w > 1):
        Head_transposed_converted = strided_head(HEAD, s_h, s_w)
    else:
        Head_transposed_converted = HEAD

    strided_head_n, strided_head_c1, strided_head_h, strided_head_w, strided_head_c0 = Head_transposed_converted.shape
    Head_transposed_converted = transpose_convert_head(Head_transposed_converted)

    s_transposed_converted = akg.tvm.create_schedule(Head_transposed_converted.op)

    pld_Head_transposed_converted = akg.tvm.placeholder(Head_transposed_converted.shape,
                                                        name="head_transposed",
                                                        dtype='float16')
    ad_attrs = {"ad_conv_enable": 1, "ad_conv_reuse_conv": 1}
    jacs = list(akg.differentiate(conv, [B], HEAD, ad_attrs, [pld_X_transposed, pld_Head_transposed_converted, None]))
    sjac = akg.tvm.create_schedule([jacs[0].op])

    op_vars = [HEAD, pld_X_transposed, pld_Head_transposed_converted, jacs[0]]
    in_n, in_c1, in_h, in_w, in_c0 = A.shape
    info = set_dims((in_c1.value * in_c0.value, in_n.value, in_h.value, in_w.value),
                    (strided_head_c1.value * strided_head_c0.value, strided_head_n.value,
                     strided_head_h.value, strided_head_w.value),
                    (0, 0), (1, 1), (1, 1),
                    tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)


    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_backward2 = akg.build(sjac, op_vars, "cce",
                                  name="conv_backward_weight",
                                  attrs={"dim": str(info)},
                                  polyhedral=True)

    return mod_backward, mod_backward2


def conv_02(fmap_shape, filter_shape, pad_, stride_, dilation_,
            tile_hh=0, tile_coco=0, tile_mm=0, tile_kk=0, tile_nn=0, bypass_l1=False,
            use_bias=False, block_size=16, conv_dtype='float16'):

    # input shape (NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape
    in_c = (in_c + block_size - 1) // block_size * block_size
    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    k_c = (k_c + block_size - 1) // block_size * block_size
    k_n = (k_n + block_size - 1) // block_size * block_size

    input_shape_nc1hwc0 = (in_n, in_c // block_size, in_h, in_w, block_size)
    in_n, _, in_h, in_w, _ = input_shape_nc1hwc0

    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, _, k_h, k_w, _ = kernel_shape_nc1hwc0
    kernel_shape_fractal = (k_c // block_size * k_h * k_w, k_n // block_size, block_size, block_size)

    # A placeholder (NC1HWCO)
    A = akg.tvm.placeholder(input_shape_nc1hwc0, dtype=conv_dtype, name="input0")
    # B_placeholder (fractal)
    B = akg.tvm.placeholder(kernel_shape_fractal, dtype=conv_dtype, name="input1")

    if use_bias:
        bias_shape_nc1hwc0 = (1, k_n // block_size, 1, 1, block_size)
        bias_name = "input2"
        bias_value = akg.tvm.placeholder(bias_shape_nc1hwc0, dtype=conv_dtype, name=bias_name)
    else:
        bias_name = 'None'
        bias_value = None

    conv_forward = conv_compute_forward(fmap_shape, filter_shape, pad_, stride_, dilation_, A, B, bias_value,
                                        tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, bypass_l1,
                                        use_bias, block_size, conv_dtype)

    k_hw = k_h * k_w
    const_shift = k_hw - 1

    # B in Fractal format; result in Fractal format
    def flip_weight(B, k_c, k_hw, const_shift):
        out_shape = (B.shape[1].value * k_hw, k_c // block_size, block_size, block_size)
        B_flip = akg.tvm.compute(out_shape,
                                 lambda i0, i1, i2, i3:
                                 B[i1 * k_hw + const_shift - truncmod(i0, k_hw), floordiv(i0, k_hw), i3, i2],
                                 name=B.name + "_flipped")
        return B_flip

    # H in 5D format; result in 5D format
    def strided_head(H, s_h, s_w):
        n, c1, h, w, c0 = H.shape
        out_shape = (n, c1, (h - 1) * s_h + 1, (w - 1) * s_w + 1, c0)
        H_strided = akg.tvm.compute(out_shape,
                                    lambda i0, i1, i2, i3, i4:
                                    akg.tvm.expr.Select(akg.tvm.any(truncmod(i2, s_h) != 0, truncmod(i3, s_w) != 0),
                                                        akg.tvm.const(0.0, dtype="float16"),
                                                        H[i0, i1, floordiv(i2, s_h), floordiv(i3, s_w), i4]),
                                    name=H.name + "_strided")

        return H_strided

    # A in 5D format; result in 5D format
    def transpose_data(A):
        out_shape = (A.shape[1].value * block_size, A.shape[0].value // block_size,
                     A.shape[2].value, A.shape[3].value, block_size)

        A_transpose = akg.tvm.compute(out_shape,
                                      lambda j0, j1, j2, j3, j4:
                                      A[j1 * block_size + j4, floordiv(j0, block_size), j2, j3, truncmod(j0, block_size)],
                                      name=A.name + "_transposed")
        return A_transpose

    # Head is in 5D format; result in Fractal format
    def transpose_convert_head(Head):
        out_shape = ((Head.shape[0].value // block_size) * Head.shape[2].value * Head.shape[3].value,
                     Head.shape[1].value, block_size, block_size)
        tmp_6D_shape = (Head.shape[0].value // block_size, block_size,
                        Head.shape[1].value, Head.shape[2].value, Head.shape[3].value, block_size)
        Head_6D = akg.topi.reshape(Head, tmp_6D_shape)
        Head_6D_transpose = akg.topi.transpose(Head_6D, (0, 3, 4, 2, 5, 1))
        Head_transpose_convert = akg.topi.reshape(Head_6D_transpose, out_shape)
        return Head_transpose_convert

    HEAD = akg.tvm.placeholder(conv_forward.shape, name="Head", dtype='float16')
    Head_transposed_NCHW = (HEAD.shape[1].value * HEAD.shape[4].value, HEAD.shape[0].value,
                            HEAD.shape[2].value, HEAD.shape[3].value)
    s_h, s_w = stride_
    Head_strided_NCHW = (HEAD.shape[0].value, HEAD.shape[1].value * HEAD.shape[4].value,
                         (HEAD.shape[2].value - 1) * s_h + 1, (HEAD.shape[3].value - 1) * s_w + 1)

    A_transposed_NCHW = (in_c, in_n, in_h, in_w)
    K_flip_rot_NCHW = (k_c, k_n, k_h, k_w)

    Head_transposed_converted = transpose_convert_head(HEAD)
    pld_Head_transposed_converted = akg.tvm.placeholder(Head_transposed_converted.shape,
                                                    name="Head_trans_fractal", dtype=conv_dtype)
    A_transposed = transpose_data(A)
    pld_A_transposed = akg.tvm.placeholder(A_transposed.shape, name="A_trans", dtype=conv_dtype)

    info = dim.Dim()
    info.setdim(index=0, axis=0, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=1, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=2, tilel1=1, tilel0=1)
    info.setdim(index=0, axis=3, tilel1=1, tilel0=1)

    B_flip = flip_weight(B, k_c, k_hw, const_shift)
    pld_B_flipped = akg.tvm.placeholder(B_flip.shape, name="B_flip", dtype=conv_dtype)

    s_flipped = akg.tvm.create_schedule(B_flip.op)
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_weight_flipped = akg.build(s_flipped, [B, B_flip], "cce", name=B.name + "_flipped",
                                      attrs={"dim": str(info)}, polyhedral=True)

    s_transposed_converted = akg.tvm.create_schedule(Head_transposed_converted.op)


    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_head_transposed_converted = akg.build(s_transposed_converted, [HEAD, Head_transposed_converted],
                                                 "cce", name="H_trans_converted",
                                                  attrs={"dim": str(info)},
                                                  polyhedral=True)

    Head_strided = strided_head(HEAD, s_h, s_w)
    pld_Head_strided = akg.tvm.placeholder(Head_strided.shape, name="Head_trans_5D", dtype=conv_dtype)

    s_strided = akg.tvm.create_schedule(Head_strided.op)
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_head_strided = akg.build(s_strided, [HEAD, Head_strided],
                                    "cce", name="H_strided", attrs={"dim": str(info)}, polyhedral=True)

    s_transposed = akg.tvm.create_schedule(A_transposed.op)


    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_transposed = akg.build(s_transposed, [A, A_transposed], "cce",
                                   name="A_transposed", attrs={"dim": str(info)}, polyhedral=True)

    ad_attrs = {"ad_conv_enable": 1, "ad_conv_reuse_conv": 1}
    jacs = list(akg.differentiate(conv_forward, [A], HEAD, ad_attrs, [pld_Head_strided, pld_B_flipped, None]))
    info = set_dims(Head_strided_NCHW, (k_c, k_n, k_h, k_w), (k_h - 1, k_w - 1), (1, 1), (1, 1),
                    tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)

    sjac = akg.tvm.create_schedule([jacs[0].op])
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_AD_data = akg.build(sjac, [pld_Head_strided, pld_B_flipped, jacs[0]], "cce",
                                name="conv_AD_data", attrs={"dim": str(info)}, polyhedral=True)


    conv_data = conv_compute_forward(Head_strided_NCHW, K_flip_rot_NCHW,
                                     (k_h - 1, k_h - 1, k_w - 1, k_w - 1), (1, 1), (1, 1),
                                     pld_Head_strided, pld_B_flipped, None,
                                     tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, bypass_l1,
                                     use_bias, block_size, conv_dtype)

    info = set_dims(Head_strided_NCHW, (k_c, k_n, k_h, k_w), (k_h - 1, k_w - 1), (1, 1), (1, 1),
                    tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)

    s_data = akg.tvm.create_schedule(conv_data.op)

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_data = akg.build(s_data, [pld_Head_strided, pld_B_flipped, conv_data], "cce",
                             name="conv_data", attrs={"dim": str(info)}, polyhedral=True)

    ad_attrs = {"ad_conv_enable": 1, "ad_conv_reuse_conv": 1}
    jacs = list(akg.differentiate(conv_forward, [B], HEAD, ad_attrs, [pld_A_transposed, pld_Head_transposed_converted, None]))
    info = set_dims(A_transposed_NCHW, Head_transposed_NCHW, (0, 0), (1, 1), (s_h, s_w),
                    tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)

    sjac = akg.tvm.create_schedule([jacs[0].op])
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_AD_weight = akg.build(sjac, [pld_A_transposed, pld_Head_transposed_converted, jacs[0]], "cce",
                                  name="conv_AD_weight", attrs={"dim": str(info)}, polyhedral=True)

    conv_weight = conv_compute_forward(A_transposed_NCHW, Head_transposed_NCHW,
                                       (0, 0, 0, 0), (1, 1), (s_h, s_w),
                                       pld_A_transposed, pld_Head_transposed_converted, None,
                                       tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, bypass_l1,
                                       use_bias, block_size, conv_dtype)

    info = set_dims(A_transposed_NCHW, Head_transposed_NCHW, (0, 0), (1, 1), (s_h, s_w),
                    tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, block_size)

    s_weight = akg.tvm.create_schedule(conv_weight.op)

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod_weight = akg.build(s_weight, [pld_A_transposed, pld_Head_transposed_converted, conv_weight], "cce",
                               name="conv_weight", attrs={"dim": str(info)}, polyhedral=True)

    return mod_AD_data, mod_AD_weight, mod_transposed, mod_head_transposed_converted, mod_head_strided, mod_weight_flipped
