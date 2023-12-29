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
fused operator dsl function: fused_bn_follow_relu_avgpool
ResNet50 fused computation. 483 in XLA patterns
"""
from __future__ import absolute_import
import akg.topi as topi
import akg.utils as utils
from tests.common.test_op.resnet.fused_bn_follow_relu import fused_bn_follow

def fused_bn_follow_relu_avgpool(data0, data1, data2, data3, data4, data5, layout='NHWC', out_dtype='float16', target=utils.CUDA):
    """
    input:
    data: length is 6
    data0: tensor1 after bn_double_relu
    data1-6: bn parameters for conv2d tensor2
    layout: only (N, H, W, C), (N, C, H, W) supported
    out_dtype: float16

    output:
    avg-pooling( max(batch-normalized tensor1 + batch-normalized tensor2,  0) )
    """
    if layout == 'NCHW':
        data0 = topi.transpose(data0, (0, 2, 3, 1))
        data5 = topi.transpose(data5, (0, 2, 3, 1))
    elif layout != 'NHWC':
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    n, h, w, c = data0.shape
    inter_dtype = 'float32'
    add0 = fused_bn_follow(data1, data2, data3, data4, data5)
    add0 = topi.cast(add0, data0.dtype)
    add1 = topi.add(data0, add0)
    output = topi.maximum(add1, 0)
    output = topi.cast(output, inter_dtype)
    output = topi.sum(output, axis=(1, 2))
    output = topi.divide(output, h * w)
    output = topi.cast(output, out_dtype)

    return output
