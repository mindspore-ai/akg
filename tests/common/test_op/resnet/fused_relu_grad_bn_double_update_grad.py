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
         ResNet50 fused_computation. 235 in XLA patterns  """
from __future__ import absolute_import
import akg.topi as topi

def fused_relu_grad_bn_double_update_grad(data_1, data_2, data_3, data_4, data_5, data_6, data_7, layout='NHWC'):
    transform_list = [data_2, data_4, data_5, data_6, data_7]
    for i in transform_list:
        if layout == "NCHW":
            i = topi.transpose(i, axes=(0, 2, 3, 1))
        elif layout != "NHWC":
            raise NotImplementedError( 'Layout not supported {} '.format(layout))

    data_tmp1 = topi.full_like(data_7, 0.0)
    data_tmp2 = topi.greater(data_7, data_tmp1)
    data_tmp3 = topi.add(data_5, data_6)
    data_tmp4 = topi.where(data_tmp2, data_tmp3, data_tmp1)
    data_tmp5 = topi.cast(data_tmp4, 'float32')
    data_tmp7 = topi.sum(data_tmp5, axis=(0, 1, 2))

    n, h, w, c = data_7.shape
    data_tmp8 = topi.cast(data_2, 'float32')
    data_tmp9 = topi.full_like(data_tmp7, 1.0/(n*h*w))
    data_tmp10 = topi.multiply(data_1, data_tmp9)
    data_tmp11 = topi.broadcast_to(data_tmp10, data_tmp8.shape)
    data_tmp12 = topi.subtract(data_tmp8, data_tmp11)
    data_tmp13 = topi.multiply(data_tmp5, data_tmp12)
    data_tmp15 = topi.sum(data_tmp13, axis=(0, 1, 2))

    data_tmp16 = topi.cast(data_4, 'float32')
    data_tmp17 = topi.multiply(data_3, data_tmp9)
    data_tmp18 = topi.broadcast_to(data_tmp17, data_tmp16.shape)
    data_tmp19 = topi.subtract(data_tmp16, data_tmp18)
    data_tmp20 = topi.multiply(data_tmp5, data_tmp19)
    data_tmp22 = topi.sum(data_tmp20, axis=(0, 1, 2))

    return [data_tmp7, data_tmp15, data_tmp22]
