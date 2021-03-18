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

""" fused operator dsl function: fused_is_finite
         ResNet50 fused_computation. 82 in XLA patterns  """
from __future__ import absolute_import
import akg.topi as topi

def fused_is_finite(data, layout='NHWC'):
    """
    fused_is_finite.

    Args:
        input: tvm.tensor.Tensor.

    Returns:
        ret.
    """
    if layout == "NCHW":
        data = topi.transpose(data, axes=(0, 2, 3, 1))
    elif layout != "NHWC":
        raise NotImplementedError('Layout not supported {} '.format(layout))
    data_isfinite = topi.isfinite(data)
    n, h, w, c = data_isfinite.shape
    data_out = topi.all(data_isfinite, axis=(0, 1, 2, 3))
    return data_out
    