# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
# limitations under the License

"""
fused operator dsl function: fused_relu_grad_bn_double_reduce_grad
ResNet50 fused computation. 227 in XLA patterns
"""

from __future__ import absolute_import
import akg.tvm as tvm
import akg.topi as topi
from topi import tag

def fused_relu_grad_bn_double_reduce_grad(data0, data1, data2, data3, data4, data5, data6, data7, data8,
                           data9, data10, data11, data12, data13, data14, data15, layout="NHWC", out_dtype="float16"):
    
    if layout == 'NCHW':
        data5 = topi.transpose(data5, (0, 2, 3, 1))
        data9 = topi.transpose(data9, (0, 2, 3, 1))
        data13 = topi.transpose(data13, (0, 2, 3, 1))
        data14 = topi.transpose(data14, (0, 2, 3, 1))
        data15 = topi.transpose(data15, (0, 2, 3, 1))
    elif layout != 'NHWC':
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))
    
    inter_dtype = "float32"
    n, h, w, c = data5.shape
    scale = n * h * w

    mul = topi.multiply(data2, data3)
    mul1221 = topi.divide(mul, scale)

    # ReluGrad
    zero = tvm.const(0, data15.dtype)
    add = topi.add(data13, data14)
    addgrad = tvm.compute(add.shape, lambda *i: tvm.if_then_else(data15(*i) >= zero, add(*i), zero), tag=tag.INJECTIVE)
    addgrad = topi.cast(addgrad, inter_dtype)
    mul3283 = topi.multiply(scale, addgrad)
    sub1159 = topi.subtract(mul3283, data6)

    data5_cast = topi.cast(data5, inter_dtype)
    mul2372 = topi.divide(data4, scale)
    sub631 = topi.subtract(data5_cast, mul2372)
    mul1220 = topi.multiply(sub631, data1)
    div = topi.divide(mul1220, data0)
    sub271 = topi.subtract(sub1159, div)
    mul1218 = topi.multiply(mul1221, sub271)
    mul1218_cast = topi.cast(mul1218, out_dtype)

    mul1231 = topi.multiply(data11, data12)
    mul1230 = topi.divide(mul1231, scale)
    data9_cast = topi.cast(data9, inter_dtype)
    mul2364 = topi.divide(data8, scale)
    sub625 = topi.subtract(data9_cast, mul2364)
    mul1229 = topi.multiply(data10, sub625)

    div272 = topi.divide(mul1229, data7)
    sub272 = topi.subtract(sub1159, div272)
    mul1228 = topi.multiply(mul1230, sub272)
    mul1228_cast = topi.cast(mul1228, out_dtype)

    if layout == "NCHW":
        mul1218_cast = topi.transpose(mul1218_cast, (0, 3, 1, 2))
        mul1228_cast = topi.transpose(mul1228_cast, (0, 3, 1, 2))
    
    return [mul1218_cast, mul1228_cast]



