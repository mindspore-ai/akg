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

"""operator dsl function: bessel_i1e"""
from akg import topi
import akg.tvm
import akg.utils as utils
from akg.ops.math import abs, cast, mul, neg, rsqrt, exp, divide
from akg.ops.math.ascend import Sign
from akg.utils.kernel_exec import product_is_mini

ITR_BEFORE = (0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658773, 0.00301532, 0.00032411)
ITR_AFTER = (0.39894228, -0.03988024, -0.00362018,
             0.00163801, -0.01031555, 0.02282967,
             -0.02895312, 0.01787654, -0.00420059)
LEN_BEFORE = 7
LEN_AFTER = 9
CONST_LIMIT = 15.0/4


def _before_res_compute(abs_data):
    """
    compute bessel_i1e for abs value of data less than or equal to 3.75

    Algrithm:
    t = x / 3.75
    I1(x) = e^-|x|*x*(0.5 + 0.87890594t^2 + 0.51498869t^4 + 0.15084934t^6
                    + 0.02658773t^8 + 0.00301532t^10 + 0.00032411t^12)
    """

    data = topi.multiply(abs_data, 1.0 / CONST_LIMIT)
    data_square = mul(data, data, target=utils.CCE)
    before_res = topi.multiply(data_square, ITR_BEFORE[LEN_BEFORE - 1])
    before_res = topi.add(before_res, ITR_BEFORE[LEN_BEFORE - 2])
    for iter_number in ITR_BEFORE[LEN_BEFORE-3::-1]:
        before_res = mul(before_res, data_square, target=utils.CCE)
        before_res = topi.add(before_res, iter_number)
    exp_value = exp(neg(abs_data, target=utils.CCE), target=utils.CCE)
    before_res = mul(before_res, exp_value, target=utils.CCE)
    before_res = mul(before_res, abs_data, target=utils.CCE)
    return before_res


def _after_res_compute(abs_data):
    """
    compute bessel_i1e for abs value of data greater than or equal to 3.75

    Algrithm:
    t = 3.75 / x
    I1(x) = (1 / sqrt(x))*(0.39894228 - 0.03988024t - 0.00362018t^2
                           + 0.00163801t^3 - 0.01031555t^4 + 0.02282967t^5
                           - 0.02895312t^6 + 0.01787654t^7 - 0.00420059t^8)
    """
    broad_const_limit = akg.lang.ascend.broadcast(akg.tvm.const(
                        CONST_LIMIT, abs_data.dtype), abs_data.shape)
    data = divide(broad_const_limit, abs_data, target=utils.CCE)
    after_res = topi.multiply(data, ITR_AFTER[LEN_AFTER - 1])
    after_res = topi.add(after_res, ITR_AFTER[LEN_AFTER - 2])
    for iter_number in ITR_AFTER[LEN_AFTER-3::-1]:
        after_res = mul(after_res, data, target=utils.CCE)
        after_res = topi.add(after_res, iter_number)
    abs_data_rsqrt = rsqrt(abs_data, target=utils.CCE)
    after_res = mul(after_res, abs_data_rsqrt, target=utils.CCE)
    return after_res


def _bessel_i1e_compute(input_data):
    """bessel i1e compute"""

    shape = utils.get_shape(input_data)
    dtype = input_data.dtype

    # chose the type of data in begin
    if dtype == "float16":
        input_data = cast(input_data, "float32", target=utils.CCE)

    abs_data = abs(input_data, utils.CCE)
    # compute bessel_i1e for data in (-3.75, 3.75)
    before_res = _before_res_compute(abs_data)
    # compute bessel_i1e for data in other domain
    after_res = _after_res_compute(abs_data)

    # As vcmp_lt and vsel instruction don't support fp32 on mini
    # It can be simplified by some methods, such as , "auto cast"
    if product_is_mini():
        res = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(
            abs_data[indice].astype("float16") < akg.tvm.const(CONST_LIMIT, "float16"),
            before_res[indice].astype("float16"),
            after_res[indice].astype("float16")))
        res = cast(res, "float32", target=utils.CCE)
    else:
        res = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(
              abs_data[indice] < CONST_LIMIT,
              before_res[indice], after_res[indice]))
    data_sign = Sign(input_data, target=utils.CCE)
    res = mul(res, data_sign, target=utils.CCE)
    if dtype == "float16":
        res = cast(res, "float16", target=utils.CCE)
    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def bessel_i1e(x):
    """
    The modified Bessel i1e function.

    ..math:: `I1e(x) = (e^{-|x|}) * (x/2) * ( 1 + (((x^2)/4)^1)/(1!*2!) + (((x^2)/4)^2)/(2!*3!) + ...
            + (((x^2)/4)^n)/(n!*(n+1)!)`

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor. The modified Bessel i1e function of x element-wise.
        Has the same type as x.

    """

    # check shape
    utils.check_shape(x)

    # check input tensor data_type
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)

    res = _bessel_i1e_compute(x)
    return res
