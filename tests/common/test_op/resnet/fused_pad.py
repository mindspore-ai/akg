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

""" fused operator dsl function: fused_pad ResNet50 fused_computation. 957 in XLA patterns  """
from __future__ import absolute_import
import akg.topi as topi

def fused_pad(input, pad_before, pad_after, layout='NHWC', pad_value=0.0):
    """
    fused_pad.
 
    Args:
        input : tvm.Tensor or Expr
        pad_before : list / tuple of n ints. (Pad width on each dimension to pad the before the axis begin.)
        pad_after : list / tuple of n ints. (Pad width each dimension to pad the after the axis end.)
        pad_value : float. (The value to be padded.)

    Returns
        tvm.Tensor
    """
    if layout == "NCHW":
        data = topi.transpose(data, axes=(0, 2, 3, 1))
    elif layout != "NHWC":
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))
            
    cast_after = topi.cast(input, 'float16')
    output = topi.nn.pad(cast_after, pad_before, pad_after, pad_value)
    return output