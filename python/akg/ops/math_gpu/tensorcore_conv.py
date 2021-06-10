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

"""operator dsl function: conv using tensorcore"""
import numpy as np
import akg.topi as topi
import akg.tvm as tvm
from akg.utils import validation_check as vc_util


def conv_tc(data, weight, stride=[1, 1], pad=[0, 0, 0, 0], dilation=[1, 1], out_dtype="float32", name="out"):
    batch_outer, in_h, in_w, in_c_outer = data.shape
    out_c_outer, k_h, k_w, _ = weight.shape
    pad_left, pad_right, pad_top, pad_bottom = pad
    s_h, s_w = stride
    o_h = (in_h + pad_top + pad_bottom - k_h) // s_h + 1
    o_w = (in_w + pad_left + pad_right - k_w) // s_w + 1

    has_pad = not(pad_left == 0 and pad_right ==
                  0 and pad_top == 0 and pad_bottom == 0)

    if has_pad:
        data_pad = tvm.compute(
            (batch_outer, in_h+pad_top+pad_bottom,
             in_w+pad_left+pad_right, in_c_outer),
            lambda n, h, w, i: tvm.if_then_else(
                tvm.all(h >= pad_top, h - pad_bottom < in_h,
                        w >= pad_left, w - pad_right < in_w),
                data[n, h-pad_top, w - pad_left, i],
                tvm.const(0.0, "float16"),
            ),
            name="Pad",
        )
    else:
        data_pad = data

    rc = tvm.reduce_axis((0, in_c_outer), name="rc")
    rh = tvm.reduce_axis((0, k_h), name="rh")
    rw = tvm.reduce_axis((0, k_w), name="rw")

    if out_dtype == "float32":
        out = tvm.compute(
            (batch_outer, o_h, o_w, out_c_outer),
            lambda n, h, w, o: tvm.sum(
                data_pad[n, (h * s_h + rh), (w * s_w + rw), rc].astype("float32") *
                weight[o, rh, rw, rc].astype("float32"),
                axis=[rc, rh, rw]),
            name=name
        )
    else:
        out = tvm.compute(
            (batch_outer, o_h, o_w, out_c_outer),
            lambda n, h, w, o: tvm.sum(
                data_pad[n, (h * s_h + rh), (w * s_w + rw), rc] *
                weight[o, rh, rw, rc],
                axis=[rc, rh, rw]),
            name=name
        )

    return out
