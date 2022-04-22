# Copyright 2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: conv2d"""
import akg.topi as topi
from akg.topi.util import get_const_tuple
import akg.tvm as tvm
from .conv_utils import get_channel_inners, pack_data, unpack_nchwc_to_nchw


def conv2d_nchwc(data, weight, stride, pad, dilation,
                 out_dtype="float32", output_layout="NCHWc", c_inners=None, target="llvm"):
    """Conv2D impl for NCHW/NCHWc layout.
    We use direct algorithm so that algo-inner layout is NCHWc.
    Func "conv2d_nchw" also support NCHWc format directly.
    """

    # default params
    if stride == None:
        stride = [1, 1]
    if pad == None:
        pad = [0, 0, 0, 0]
    if dilation == None:
        dilation = [1, 1]
    if c_inners == None:
        c_inners = [-1, -1]

    if len(data.shape) == 4:
        # data layout = NCHW
        batch, in_channel, i_h, i_w = get_const_tuple(data.shape)
        out_channel, in_channel, k_h, k_w = get_const_tuple(weight.shape)
    elif len(data.shape) == 5 or len(data.shape) == 6:
        # data layout = NCHWc/NCHW[x]c
        batch, ic_outer, i_h, i_w, ic_inner = get_const_tuple(data.shape)
        oc_outer, ic_outer, k_h, k_w, _, oc_inner = get_const_tuple(
            weight.shape)
        in_channel = ic_outer * ic_inner
        out_channel = oc_outer * oc_inner
    else:
        raise ValueError(
            "length of data shape should be 4~6, now is {}".format(len(data.shape)))

    pad_top, pad_bottom, pad_left, pad_right = pad
    s_h, s_w = stride
    d_h, d_w = dilation
    k_h_d = (k_h - 1) * d_h + 1
    k_w_d = (k_w - 1) * d_w + 1
    o_h = (i_h + pad_top + pad_bottom - k_h_d) // s_h + 1
    o_w = (i_w + pad_left + pad_right - k_w_d) // s_w + 1

    if len(data.shape) == 4:
        # layout transform: NCHW -> NCHWc
        ic_inner, oc_inner = get_channel_inners(c_inners[0], c_inners[1],
                                                in_channel, out_channel, target)
        ic_outer = in_channel // ic_inner
        oc_outer = out_channel // oc_inner
        data, weight = pack_data(data, weight, ic_inner, oc_inner)

    out_shape = (batch, oc_outer, o_h, o_w, oc_inner)

    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        data_pad = data
    else:
        data_pad = topi.nn.pad(data, [0, 0, pad_top, pad_left, 0], [
                               0, 0, pad_bottom, pad_right, 0], 0.0,)

    ic_out = tvm.reduce_axis((0, ic_outer), name="ic_out")
    ic_in = tvm.reduce_axis((0, ic_inner), name="ic_in")
    kh = tvm.reduce_axis((0, k_h), name="kh")
    kw = tvm.reduce_axis((0, k_w), name="kw")

    out = tvm.compute(out_shape,
                      lambda batch, oc_out, oh, ow, oc_in: tvm.sum(
                          data_pad[batch,
                                   ic_out,
                                   oh * s_h + kh * d_h,
                                   ow * s_w + kw * d_w,
                                   ic_in,
                                   ].astype(out_dtype) *
                          weight[oc_out,
                                 ic_out,
                                 kh,
                                 kw,
                                 ic_in,
                                 oc_in,
                                 ].astype(out_dtype),
                          axis=[ic_out, kh, kw, ic_in]),
                      name="conv2d_nchwc")

    if output_layout == "NCHW":
        out = unpack_nchwc_to_nchw(out, out_dtype)

    return out
