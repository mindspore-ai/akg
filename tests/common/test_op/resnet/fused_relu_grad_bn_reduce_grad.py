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

""" fused operator dsl function: fused_relu_grad_bn_reduce_grad
         ResNet50 fused_computation. 215 in XLA patterns  """
from __future__ import absolute_import
import akg.topi as topi

def fused_relu_grad_bn_reduce_grad(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, layout='NHWC'):
    """
    fused_relu_grad_bn_reduce_grad.

    Args:
        data_1~data_9: tvm.tensor.Tensor.
        layout: input layout, only 'NCHW', 'NHWC' supported

    Returns:
        tvm.tensor.Tensor.
    """
    transform_list = [data_7, data_8, data_9]
    for i in transform_list:
        if layout == "NCHW":
            i = topi.transpose(i, axes=(0, 2, 3, 1))
        elif layout != "NHWC":
            raise NotImplementedError(
                'Layout not supported {} '.format(layout))
 
    data_tmp1 = topi.multiply(data_4, data_5)
    n, h, w, c = data_9.shape
    data_tmp2 = topi.full_like(data_tmp1, 1.0/(n*h*w))
    data_tmp3 = topi.multiply(data_tmp1, data_tmp2)

    data_tmp5 = topi.full_like(data_9, 0.0)
    data_tmp6 = topi.greater(data_9, data_tmp5)

    data_tmp7 = topi.where(data_tmp6, data_8, data_tmp5)

    data_tmp8 = topi.cast(data_tmp7, 'float32')
    data_tmp9 = topi.full_like(data_tmp8, n*h*w)
    data_tmp10 = topi.multiply(data_tmp8, data_tmp9)
    data_tmp12 = topi.subtract(data_tmp10, data_3)
    data_tmp14 = topi.cast(data_7, 'float32')
    data_tmp15 = topi.multiply(data_6, data_tmp2)

    data_tmp17 = topi.subtract(data_tmp14, data_tmp15)
    data_tmp18 = topi.multiply(data_2, data_tmp17)
    data_tmp20 = topi.divide(data_tmp18, data_1)
    data_tmp21 = topi.subtract(data_tmp12, data_tmp20)
    data_tmp22 = topi.multiply(data_tmp3, data_tmp21)
    data_out = topi.cast(data_tmp22, 'float16')

    return data_out

