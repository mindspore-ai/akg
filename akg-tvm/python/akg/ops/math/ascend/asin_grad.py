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

"""operator dsl function: asin_grad"""
import akg
import akg.utils as utils
from akg import topi, tvm
from akg.utils.kernel_exec import product_is_mini


def _newton(start_value, num_to_vrsqrt):
    """Do newton's method to calculate vrsqrt."""

    x0_square = topi.multiply(start_value, start_value)
    mul_res = topi.multiply(x0_square, num_to_vrsqrt)
    mul_res = topi.multiply(mul_res, tvm.const(-1, "float32"))
    head0_tmp = topi.add(mul_res, tvm.const(3, "float32"))
    head0 = topi.multiply(head0_tmp, start_value)
    newton_res = topi.multiply(head0, tvm.const(0.5, "float32"))

    return newton_res


def _vrsqrt_newton(num_to_vrsqrt):
    """Calculate vrsqrt(num_to_vrsqrt) with newton's method."""

    start_value = topi.rsqrt(num_to_vrsqrt)

    # use newton's method 3 times
    newton_res_1 = _newton(start_value, num_to_vrsqrt)
    newton_res_2 = _newton(newton_res_1, num_to_vrsqrt)
    newton_res_3 = _newton(newton_res_2, num_to_vrsqrt)

    return newton_res_3


def _asin_grad_compute(x, dy):
    """Compute asin_grad."""

    dtype = x.dtype
    if dtype == "float16":
        x = topi.cast(x, "float32")
        dy = topi.cast(dy, "float32")

    """step 1: calculate num_to_vrsqrt = 1 - x^2"""
    data = topi.multiply(x, x)
    data = topi.multiply(data, tvm.const(-1, "float32"))
    num_to_vrsqrt = topi.add(data, tvm.const(1, "float32"))

    # step 2: calculate dy * (1 / sqrt(1 - x^2))
    if product_is_mini():
        # mini: use newton's method for high accuracy result
        res = _vrsqrt_newton(num_to_vrsqrt)
        res = topi.multiply(res, dy)
    else:
        # cloud: use vdiv for high efficiency computation
        vsqrt_res = topi.sqrt(num_to_vrsqrt)
        res = topi.divide(dy, vsqrt_res)

    if dtype == "float16":
        res = topi.cast(res, "float16")

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def asin_grad(x, dy):
    """
    Gradient for arcsin.

    .. math::
        \\frac {\\partial arcsin(x)} {\\partial x} = \\frac{1}{\\sqrt{1 - x^2}}

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.
        dy (tvm.tensor.Tensor): Tensor of same type and shape as x.

    Rerurns:
        tvm.tensor.Tensor of same type and shape as x.
    
    Supported Platforms:
        'Ascend'
    """
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.elemwise_dtype_check(x.dtype, dy.dtype)
    utils.elemwise_shape_check(x.shape, dy.shape)

    return _asin_grad_compute(x, dy)
