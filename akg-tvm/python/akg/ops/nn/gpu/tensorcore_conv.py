# Copyright 2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: conv2d using tensorcore"""
import akg.tvm as tvm
import akg.topi as topi
import akg.utils as utils

def TensorcoreConv(data, weight, stride=[1, 1], pad=[0, 0, 0, 0], dilation=[1, 1], out_dtype="float32",
                    name="out", target=utils.CUDA):
    batch, in_h, in_w, in_c = data.shape
    out_c, k_h, k_w, _ = weight.shape
    pad_top, pad_bottom, pad_left, pad_right  = pad
    s_h, s_w = stride
    d_h, d_w = dilation
    k_h_d = (k_h - 1) * d_h + 1
    k_w_d = (k_w - 1) * d_w + 1
    o_h = (in_h + pad_top + pad_bottom - k_h_d) // s_h + 1
    o_w = (in_w + pad_left + pad_right - k_w_d) // s_w + 1

    has_pad = not(pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0)

    if has_pad:
        data_pad = tvm.compute(
            (batch, in_h+pad_top+pad_bottom, in_w+pad_left+pad_right, in_c),
            lambda n, h, w, i: tvm.if_then_else(
                tvm.all(h >= pad_top, h - pad_bottom < in_h, w >= pad_left, w - pad_right < in_w),
                data[n, h - pad_top, w - pad_left, i],
                tvm.const(0.0, "float16"),
            ),
            name="Pad",
        )
    else:
        data_pad = data

    rc = tvm.reduce_axis((0, in_c), name="rc")
    rh = tvm.reduce_axis((0, k_h), name="rh")
    rw = tvm.reduce_axis((0, k_w), name="rw")

    if out_dtype == "float32":
        out = tvm.compute(
            (batch, o_h, o_w, out_c),
            lambda n, h, w, o: tvm.sum(
                data_pad[n, (h * s_h + rh * d_h), (w * s_w + rw * d_w), rc].astype("float32")
                * weight[o, rh, rw, rc].astype("float32"),
                axis=[rc, rh, rw]),
            name=name
        )
    else:
        out = tvm.compute(
            (batch, o_h, o_w, out_c),
            lambda n, h, w, o: tvm.sum(
                data_pad[n, (h * s_h + rh * d_h), (w * s_w + rw * d_w), rc]
                * weight[o, rh, rw, rc],
                axis=[rc, rh, rw]),
            name=name
        )

    return out
