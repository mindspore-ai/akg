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
fused operator dsl function: fused_bn_update
ResNet50 fused_computation.485 in XLA patterns
"""
from __future__ import absolute_import
import akg.tvm as tvm
import akg.topi as topi

def fused_bn_update(input1, input2, input3, input4, dtype, c1, c2, c3, c4):
    """
    fused operator.

    Args:
        input1 ~ input4: tvm.tensor.Tensor.
        dtype: dtype of Tensor.
        c1 ~ c4: const.

    Returns:
        Three output (list of tvm.tensor.Tensor).
    """
    const1 = tvm.const(c1, dtype)
    mul0 = topi.multiply(input2, const1)
    mul1 = topi.multiply(input1, const1)
    mul2 = topi.multiply(mul1, mul1)
    sigma2 = topi.subtract(mul0, mul2)
    const2 = tvm.const(c2, dtype)
    rsqrt_val = topi.rsqrt(topi.add(sigma2, const2))

    const3 = tvm.const(c3, dtype)
    mul3 = topi.multiply(sigma2, const3)
    sub1 = topi.subtract(input3, mul3)
    const4 = tvm.const(c4, dtype)
    data1 = topi.multiply(const4, sub1)

    sub2 = topi.subtract(input4, mul1)
    data2 = topi.multiply(const4, sub2)

    return (rsqrt_val, data1, data2)