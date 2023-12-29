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
fused operator dsl function: fused_bn_double_follow_relu
ResNet50 fused computation. 539 in XLA patterns
"""
from __future__ import absolute_import
import akg.topi as topi
import akg.utils as utils
from tests.common.test_op.resnet.fused_bn_follow_relu import fused_bn_follow

def fused_bn_double_follow_relu(data0, data1, data2, data3, data4,
                         data5, data6, data7, data8, data9, layout='NHWC', out_dtype='float16', target=utils.CUDA):
    """
    input:
    data: length is 5
    data0-4: bn parameters for conv2d tensor 1
    data5-9: bn parameters for conv2d tensor 2
    layout: only (N, H, W, C), (N, C, H, W) supported
    out_dtype: float16

    output:
    ReLU: max(batch-normalized tensor1 + batch-normalized tensor2,  0)
    """

    if layout == 'NCHW':
        data4 = topi.transpose(data4, (0, 2, 3, 1))
        data9 = topi.transpose(data9, (0, 2, 3, 1))
    elif layout != 'NHWC':
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    add0 = fused_bn_follow(data0, data1, data2, data3, data4)
    add1 = fused_bn_follow(data5, data6, data7, data8, data9)
    add0 = topi.cast(add0, out_dtype)
    add1 = topi.cast(add1, out_dtype)
    add2 = topi.add(add0, add1)
    output = topi.maximum(add2, 0)

    if layout == "NCHW":
        output = topi.transpose(output, (0, 3, 1, 2))

    return output
