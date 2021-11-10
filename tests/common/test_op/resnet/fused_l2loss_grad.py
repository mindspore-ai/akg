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

""" fused operator dsl function: fused_l2loss_grad
         ResNet50 fused_computation. 83 in XLA patterns  """
from __future__ import absolute_import
import akg.topi as topi
import akg.utils as utils

def fused_l2loss_grad(data_f16, data_f32, layout='NHWC', fill_data=4e-05, target=utils.CUDA):
    """
    fused_l2loss_grad.

    Args:
        input: tvm.tensor.Tensor.

    Returns:
        ret.
    """
    if layout == "NCHW":
        data_f16 = topi.transpose(data_f16, axes=(0,2,3,1))
    elif layout != "NHWC":
        raise NotImplementedError('Layout not supported {} '.format(layout))

    data_f16 = topi.cast(data_f16, 'float32')
    constant_tmp = topi.cast(fill_data, 'float32')
    data_constant = topi.full_like(data_f32, constant_tmp)
    data_out = topi.multiply(data_constant, data_f32)
    data_out = topi.add(data_f16, data_out)

    return data_out
