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
fused operator dsl function: fused_bn_follow_relu 
ResNet50 fused_computation. 494 in XLA patterns
"""
from __future__ import absolute_import
import akg.topi as topi

def fused_bn_follow(data0, data1, data2, data3, data4):
    """
    input:
    data: length is 5
    data0: param0 beta
    data1: param1 gamma
    data2: param2 BNupdate: xi_variance
    data3: param6 BNreduce: xi_mean
    data4: param7 xi_conv2d

    layout: (N, C, H, W)

    output:
    beta + gamma * xi_variance * ( xi -  xi_mean/(N*H*W) )
    """

    n, h, w, c = data4.shape
    const = n * h * w
    inter_dtype = 'float32'
    data4 = topi.cast(data4, inter_dtype)

    multiply0 = topi.divide(data3, const)
    multiply0 = topi.expand_dims(multiply0, axis=0, num_newaxis=3)
    multiply0 = topi.broadcast_to(multiply0, (n, h, w, c))

    subtract0 = topi.subtract(data4, multiply0)

    multiply1 = topi.multiply(subtract0, data2)
    multiply2 = topi.multiply(multiply1, data1)

    add0 = topi.add(multiply2, data0)

    return add0

def fused_bn_follow_relu(data0, data1, data2, data3, data4, layout='NHWC', out_dtype='float16'):
    """
    input:
    data0-4: bn parameters for conv2d tensor, length is 5
    data0: param0 beta
    data1: param1 gamma
    data2: param2 BNupdate: xi_variance
    data3: param6 BNreduce: xi_mean
    data4: param7 xi_conv2d, float16
    layout: only (N, H, W, C), (N, C, H, W) supported
    out_dtype: float16

    output:
    ReLU: max(batch-normalized tensor,  0)
    """
    if layout == 'NCHW':
        data4 = topi.transpose(data4, (0, 2, 3, 1))
    elif layout != 'NHWC':
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    add0 = fused_bn_follow(data0, data1, data2, data3, data4)
    add0 = topi.cast(add0, out_dtype)
    output = topi.maximum(add0, 0)

    if layout == "NCHW":
        output = topi.transpose(output, (0, 3, 1, 2))

    return output
