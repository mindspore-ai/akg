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

"""
fused operator dsl function: fused_mul_div_rsqrt_mul_isfinite_red
ResNet50 fused computation. 212 in XLA patterns
"""
from __future__ import absolute_import
import akg.topi as topi
import akg.utils as utils

def fused_mul_div_rsqrt_mul_isfinite_red(input1, input2, out_dtype="float32", target=utils.CUDA):
    """
    fused operator.

    Args:
        input1: tvm.tensor.Tensor.
        input2: tvm.tensor.Tensor.
        dtype: dtype of Tensor.

    Returns:
        list of tvm.tensor.Tensor.
    """
    mul_param1 = topi.multiply(input2, input2)
    divide_val = topi.divide(1, mul_param1)
    rsqrt_val = topi.rsqrt(divide_val)
    mul_param0 = topi.multiply(input1, rsqrt_val)
    isfinite = topi.isfinite(mul_param0)
    reduce_and = topi.all(isfinite)
    
    if mul_param0.dtype != out_dtype:
        mul_param0 = topi.cast(mul_param0, out_dtype)
        rsqrt_val = topi.cast(rsqrt_val, out_dtype)
        divide_val = topi.cast(divide_val, out_dtype)
        
    return [reduce_and, mul_param0, rsqrt_val, divide_val]