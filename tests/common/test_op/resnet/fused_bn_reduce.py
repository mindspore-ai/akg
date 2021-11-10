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
fused operator dsl function: fused_bn_reduce 
ResNet50 fused_computation. 492 in XLA patterns
"""
from __future__ import absolute_import
import akg.topi as topi
import akg.utils as utils

def fused_bn_reduce(data, layout, out_dtype, target=utils.CUDA):
    """
    input:
    data:  4-D Tensor
    layout: input layout, only 'NCHW', 'NHWC' supported
    out_dtype: "float16" or "float32"
    
    output:
    out1_sum: 1-D tensor (C), sum on the axis "C" of input
    out2_squared_sum: 1-D tensor (C), sum of  squared on the axis "C" of input
    """

    if layout == "NCHW":
        data = topi.transpose(data, axes=(0, 2, 3, 1))
    elif layout != "NHWC":
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    inter_dtype = 'float32'
    data_cast = topi.cast(data, inter_dtype)

    out1_sum = topi.sum(data_cast, axis=(0, 1, 2))
    if out1_sum.dtype != out_dtype:
        out1_sum = topi.cast(out1_sum, out_dtype)

    squared = topi.multiply(data_cast, data_cast)
    out2_squared_sum = topi.sum(squared, axis=(0, 1, 2))
    if out2_squared_sum.dtype != out_dtype:
        out2_squared_sum = topi.cast(out2_squared_sum, out_dtype)

    return [out1_sum, out2_squared_sum]
