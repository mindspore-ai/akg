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

"""operator dsl function: asin"""
import akg
import akg.utils as utils
from akg import topi, tvm
from akg.utils.kernel_exec import product_is_mini
from akg.utils.dsl_create import neg_one_const, one_const
from .sign import Sign

# BOUNDARY: sqrt(2) / 2
BOUNDARY = 0.70710678118654752440084436210485
HALF_PI = 1.5707963267948966192313216916398

# Taylor coefficient
COEF = (1.0,
        0.16666666666666666666666666666667,
        0.075,
        0.04464285714285714285714285714286,
        0.03038194444444444444444444444444,
        0.02237215909090909090909090909091,
        0.01735276442307692307692307692308,
        0.01396484375)

# TAYLOR COUNT
TAYLOR_COUNT = 7


def _newton_iter(data, init_x):
    """Do element-wise Newton compute."""
    """Newton begin:x(n+1) = x(n)*(3-a*x(n)^2)/2"""
    init_square = topi.multiply(init_x, init_x)
    newton_res = topi.multiply(init_square, data)
    newton_res = topi.multiply(newton_res, neg_one_const("float32"))
    newton_res = topi.add(newton_res, tvm.const(3, "float32"))
    newton_res = topi.multiply(newton_res, init_x)
    newton_res = topi.multiply(newton_res, tvm.const(0.5, "float32"))
    return newton_res


def _sqrt(data):
    """Calculate sqrt by using three times newton iteration(Mini) or vsqrt(Cloud)."""
    if product_is_mini():
        data_sqrt = topi.rsqrt(data)
        data_sqrt = _newton_iter(data, data_sqrt)
        data_sqrt = _newton_iter(data, data_sqrt)
        data_sqrt = _newton_iter(data, data_sqrt)
        return topi.multiply(data, data_sqrt)
    else:
        return topi.sqrt(data)


def _taylor_compute(data_x, x_square=None):
    """Do arcsinx compute use the 15th order taylor expansion when 0 <= x <= BOUNDARY."""

    if x_square is None:
        x_square = topi.multiply(data_x, data_x)
    else:
        x_square = x_square

    """asin(x) = x + 1/6*x^3 + 3/40*x^5 + 5/112*x^7 + ... + 13!!/(14!!*15)*x^15"""
    res = topi.multiply(x_square, tvm.const(COEF[TAYLOR_COUNT], "float32"))
    for temp in reversed(range(TAYLOR_COUNT)):
        res = topi.add(res, tvm.const(COEF[temp], "float32"))
        if temp == 0:
            res = topi.multiply(res, data_x)
        else:
            res = topi.multiply(x_square, res)

    return res


def _asin_compute(data_input, target):
    """Compute asin"""

    dtype = data_input.dtype
    boundary = tvm.const(BOUNDARY, "float32")

    # Change dtype to float32
    if dtype == "float16":
        data_input = topi.cast(data_input, "float32")

    # Sign mask
    data_sign = Sign(data_input, target)

    # All positive
    data1 = topi.multiply(data_input, data_sign)

    # x belongs to (0, 2^(-0.5))
    choice_1 = topi.minimum(data1, boundary)
    choice_1 = topi.subtract(choice_1, boundary)
    choice_1_floor = akg.lang.ascend.floor(choice_1)
    # the dtype of choice_1_floor is int32, need to be cast to fp32.
    if product_is_mini():
        choice_1_floor = topi.cast(choice_1_floor, "float16")
        choice_1_floor = topi.cast(choice_1_floor, "float32")
    else:
        choice_1_floor = topi.cast(choice_1_floor, "float32")
    choice_1 = topi.multiply(choice_1_floor, neg_one_const("float32"))

    taylor1 = _taylor_compute(data1)
    res_1 = topi.multiply(taylor1, choice_1)

    # x belongs to (2^(-0.5), 1)
    choice_2 = topi.subtract(one_const("float32"), choice_1)
    data2 = topi.subtract(one_const("float32"), topi.multiply(data1, data1))
    data2_sqrt = _sqrt(data2)

    taylor2 = _taylor_compute(data2_sqrt, data2)

    res_2 = topi.multiply(taylor2, neg_one_const("float32"))
    res_2 = topi.add(res_2, tvm.const(HALF_PI, "float32"))
    res_2 = topi.multiply(res_2, choice_2)

    # Restore sign
    res_1 = topi.add(res_1, res_2)
    res_1 = topi.multiply(res_1, data_sign)

    # Restore dtype
    if dtype == "float16":
        res_1 = topi.cast(res_1, "float16")

    return res_1


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def asin(x, target=utils.CCE):
    """
    Computes the trignometric inverse sine of `x` element-wise.

    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to [-1, -sqrt(2)/2)
              | the 15th order taylor expansion, x belongs to [-sqrt(2)/2, sqrt(2)/2)
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to [sqrt(2)/2, 1]

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.

    Rerurns:
        tvm.tensor.Tensor of same type and shape as x.
    
    Supported Platforms:
        'Ascend'
    """
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(x.shape)

    return _asin_compute(x, target)
