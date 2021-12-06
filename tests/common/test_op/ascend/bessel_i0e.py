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

"""operator dsl function: bessel_i0e"""
from akg import topi
import akg.tvm
import akg.utils as utils
from akg.ops.math import Abs, Cast, Mul, Neg, Rsqrt, Exp, Divide, Minimum

# const value
ITR_BEFORE = (1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813)
ITR_AFTER = (0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281,
             -0.02057706, 0.02635537, -0.01647633, 0.00392377)
LEN_BEFORE = 7
LEN_AFTER = 9
CONST_LIMIT = 15.0 / 4


def _bessel_i0e_compute(input_data):
    """bessel i0e compute"""

    shape_input = input_data.shape
    dtype_input = input_data.dtype

    # chose the type of data in begin
    if dtype_input == "float16":
        input_data = Cast(input_data, "float32", target=utils.CCE)
    abs_data = Abs(input_data, target=utils.CCE)

    # compute bessel_i0e for data in (-3.75, 3.75)
    # t = |x| / 3.75
    # I0e = e^-|x|(1 + 3.5156229t^2 + 3.0899424t^4 + 1.2067492t^6 + 0.2659732t^8
    #       + 0.0360768t^10 + 0.0045813t^12)), |x| <= 3.75
    broad_const_limit = akg.lang.ascend.broadcast(akg.tvm.const(CONST_LIMIT, "float32"), shape_input)
    before_abs_data = Minimum(abs_data, broad_const_limit)
    data = topi.multiply(before_abs_data, 1.0 / CONST_LIMIT)
    square_data = Mul(data, data, target=utils.CCE)
    before_res = topi.multiply(square_data, ITR_BEFORE[LEN_BEFORE - 1])
    before_res = topi.add(before_res, ITR_BEFORE[LEN_BEFORE - 2])
    for iter_number in ITR_BEFORE[LEN_BEFORE-3::-1]:
        before_res = Mul(before_res, square_data, target=utils.CCE)
        before_res = topi.add(before_res, iter_number)
    exp_data = Exp(Neg(before_abs_data, target=utils.CCE), target=utils.CCE)
    before_res = Mul(before_res, exp_data, target=utils.CCE)

    # compute bessel_i0e for data in other domain
    # t = |x| / 3.75
    # I0e(x) = (1 / sqrt(|x|))*(0.39894228 + 0.01328592t^-1 + 0.00225319t^-2 + -0.00157565t^-3
    #           + 0.00916281t^-4 + -0.02057706t^-5 + 0.02635537t^-6 + -0.01647633t^-7
    #           + 0.00392377t^-8), |x| >= 3.75
    data = Divide(broad_const_limit, abs_data, target=utils.CCE)
    after_res = topi.multiply(data, ITR_AFTER[LEN_AFTER - 1])
    after_res = topi.add(after_res, ITR_AFTER[LEN_AFTER - 2])
    for iter_number in ITR_AFTER[LEN_AFTER-3::-1]:
        after_res = Mul(after_res, data, target=utils.CCE)
        after_res = topi.add(after_res, iter_number)
    rsqrt_data = Rsqrt(abs_data, target=utils.CCE)
    after_res = Mul(after_res, rsqrt_data, target=utils.CCE)
    after_res = Minimum(before_res, after_res, target=utils.CCE)

    # chose the type of data in end
    if dtype_input == "float16":
        after_res = Cast(after_res, "float16", target=utils.CCE)

    return after_res


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def bessel_i0e(x, target=utils.CCE):
    """
    The modified Bessel i0e function.

    ..math:: `I0e(x) = (e^{-|x|}) * (1 + ( (x/2) / (1!) )^2 + ((x/2)^2 / (2!))^2 + ... +
                      ((x/2)^n / (n!)) ^2)`

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor. The modified Bessel i0e function of x element-wise.
        Has the same type as x.

    """
    # check shape
    utils.check_shape(x)

    # check input tensor data_type
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)

    res = _bessel_i0e_compute(x)
    return res
