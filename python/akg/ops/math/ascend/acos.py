# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""acos"""

import math
import akg.tvm
import akg.utils as utils


def factorial_with_step2(n):
    """Calculate factorial"""
    res = 1
    while (n > 0):
        res *= n
        n -= 2
    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def acos(x):
    """
    Compute acos.

    acos(x) = pi/2-x(1 + x^2(1/6 + x^2(3/40 + x^2(15/336 + 105/3456*x^2(...)))))

    Args:
        x (tvm.tensor.Tensor): tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, same type and shape as x.
    
    Supported Platforms:
        'Ascend'
    """
    dtype = x.dtype

    utils.check_shape(x.shape)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    half_pi = akg.tvm.const(math.pi / 2, dtype=dtype)
    coefficient = [
        akg.tvm.const(factorial_with_step2(2 * (i - 1) - 1) / (factorial_with_step2(2 * (i - 1)) * (2 * (i - 1) + 1)),
                      dtype=dtype) for i in range(1, 11)]
    input_square = akg.tvm.compute(x.shape, lambda *index: x(*index) * x(*index), name="input_square")
    mid = akg.tvm.compute(x.shape, lambda *index: input_square(*index) * coefficient[len(coefficient) - 1], name="mid")
    for i in range(1, len(coefficient) - 1):
        name = "" .join( "mid_res{}".format(str(len(coefficient) - 1 - i)))
        mid = akg.tvm.compute(x.shape,
                              lambda *index, ite_i=i: input_square(*index) *
                              (coefficient[len(coefficient) - 1 - ite_i] + mid(*index)),
                              name=name)
    mid = akg.tvm.compute(x.shape, lambda *index: x(*index) * (coefficient[0] + mid(*index)), name="mid_res1")
    res = akg.tvm.compute(x.shape, lambda *index: half_pi - mid(*index), name="res")

    return res
