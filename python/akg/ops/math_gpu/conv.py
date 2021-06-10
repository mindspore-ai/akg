# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function: batch_matmul"""
import numpy as np
import akg.topi as topi
import akg.tvm as tvm
from akg.utils import validation_check as vc_util

def conv(data, weight, stride=[1,1], pad=[0,0,0,0], dilation=[1,1], name="out"):
    batch, in_c, in_h, in_w = data.shape
    out_c, in_c, k_h, k_w = weight.shape
    pad_left, pad_right, pad_top, pad_bottom = pad
    s_h, s_w = stride
    o_h = (in_h + pad_top + pad_bottom - k_h) // s_h + 1
    o_w = (in_w + pad_left + pad_right - k_w) // s_w + 1
    out_shape = (batch, out_c, o_h, o_w)

    data_pad = topi.nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right], 0.0)

    rc = tvm.reduce_axis((0, in_c), name="rc")
    rh = tvm.reduce_axis((0, k_h), name="rh")
    rw = tvm.reduce_axis((0, k_w), name="rw")

    out = tvm.compute(out_shape,
                      lambda n,c,h,w: tvm.sum(data_pad[n, rc, h * s_h + rh, w * s_w + rw]*weight[c, rc, rh, rw],
                                              axis=[rc, rh, rw]), name=name)
    # use for relu condition
    # out = tvm.compute(out.shape, lambda *i: tvm.max(out(*i), tvm.const(0, out.dtype)), name="relu")
    return out

def conv_fusion(data, weight1, weight2, stride1=[1,1], stride2=[1,1], pad1=[0,0,0,0], pad2=[0,0,0,0], dilation1=[1,1], dilation2=[1,1]):
    data2 = conv(data, weight1, stride1, pad1, dilation1)
    out = conv(data2, weight2, stride2, pad2, dilation2, "out2")
    return out
