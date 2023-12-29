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

"""operator dsl function: acosh_grad"""

from akg import tvm
from akg import topi
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini


def _sinh_taylor(x):
    """compute sinh with taylor method"""

    sinh_2x_times = 3
    dtype = x.dtype

    def _sinh_taylor_compute(x):
        """sinh(x) value is x * (1 + x^2( 1/3! + x^2(1/5! + x^2/7!)))"""
        taylor_params = [tvm.const(0.1666666666666666666666666666666666, dtype),
                         tvm.const(0.0083333333333333333333333333333333, dtype),
                         tvm.const(0.0001984126984126984126984126984126, dtype)]
        x_square = topi.multiply(x, x)
        sinh_taylor = tvm.compute(x.shape, lambda *indice: x(*indice) * (1 + x_square(*indice)*(
                                  taylor_params[0] + x_square(*indice)*(
                                      taylor_params[1] + x_square(*indice)*taylor_params[2]))),
                                  name="sinh_taylor")
        return sinh_taylor

    def _sinh_2x(sinh_x):
        """sinh(2x) = 2*sinh(x)*sqrt(sinh(x)^2+1)"""
        sinh_x_square = topi.multiply(sinh_x, sinh_x)
        sinh_x_square_add_one = topi.add(sinh_x_square, 1)
        sqrt_value = topi.sqrt(sinh_x_square_add_one)
        sinh_x_mul_sqrt_value = topi.multiply(sinh_x, sqrt_value)
        sinh_2x = topi.multiply(2, sinh_x_mul_sqrt_value)
        return sinh_2x

    # First, shrink the value of x and then use the double angle formula to calculate
    x_small = topi.multiply(x, tvm.const(1/(2 ** sinh_2x_times), dtype))
    res = _sinh_taylor_compute(x_small)
    for i in range(sinh_2x_times):
        res = _sinh_2x(res)
    return res


@utils.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (str, type(None)))
def acosh_grad(y, dy):
    """
    Gradient for acosh.

    Note:
        dx = dy * 1/sinh(y)

    Args:
        y (tvm.tensor.Tensor): tensor of type float16, float32.
        dy (tvm.tensor.Tensor): same type and shape as y.

    Returns:
        tvm.tensor.Tensor, same type and shape as y.
    
    Supported Platforms:
        'Ascend'
    """

    # mini product just used to infer
    if product_is_mini():
        raise RuntimeError("The mini product does not support the acosh_grad operator")

    dtype = y.dtype
    utils.ops_dtype_check(y.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.elemwise_dtype_check(dtype, dy.dtype)
    utils.check_shape(y.shape)
    utils.elemwise_shape_check(y.shape, dy.shape)

    if dtype == "float16":
        y = topi.cast(y, "float32")
        dy = topi.cast(dy, "float32")

    # If we use sinh(y) = (exp(y) - exp(-y))/2 directly, there will be some precision problems
    # For example, as dx = dy/sinh(y), if we use exp directly, when exp(y) and exp(-y) are close,
    # the small precision error of exp calculation will be greatly enlarged in the final result
    sinh_y = _sinh_taylor(y)
    dx = topi.divide(dy, sinh_y)

    if dx.dtype != dtype:
        dx = topi.cast(dx, dtype)
    attrs = {"enable_auto_inline": False}
    return dx, attrs
