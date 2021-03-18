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
# limitations under the License.

"""
fused operator dsl function: fused_bn_reduce_grad
ResNet50 fused computation. 245 in XLA patterns
"""
from __future__ import absolute_import
import akg.topi as topi

def fused_bn_reduce_grad(data0, data1, data2, data3, data4, data5, data6, data7, layout='NHWC', out_dtype='float16'):
    
    if layout == 'NCHW':
        data3 = topi.transpose(data3, (0, 2, 3, 1))
        data7 = topi.transpose(data7, (0, 2, 3, 1))
    elif layout != 'NHWC':
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    n, h, w, c = data3.shape
    const = n * h * w
    inter_dtype = 'float32'
    out1 = topi.multiply(data4, data5)
    out1 = topi.divide(out1, const)
    out1 = topi.expand_dims(out1, axis=0, num_newaxis=3)
    out1 = topi.broadcast_to(out1, (n, h, w, c))

    data3 = topi.cast(data3, inter_dtype)
    data2 = topi.expand_dims(data2, axis=0, num_newaxis=3)
    data2 = topi.broadcast_to(data2, (n, h, w, c))
    out2 = topi.multiply(data3, const)
    out2 = topi.subtract(out2, data2)

    data1 = topi.expand_dims(data1, axis=0, num_newaxis=3)
    data1 = topi.broadcast_to(data1, (n, h, w, c))
    data7 = topi.cast(data7, inter_dtype)
    out3 = topi.divide(data6, const)
    out3 = topi.subtract(data7, out3)
    out3 = topi.multiply(data1, out3)
    out3 = topi.divide(out3, data0)

    output = topi.subtract(out2, out3)
    output = topi.multiply(output, out1)

    output = topi.cast(output, out_dtype)

    if layout == "NCHW":
        output = topi.transpose(output, (0, 3, 1, 2))

    return output
