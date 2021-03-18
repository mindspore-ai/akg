# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function: tan"""
import akg.lang.cce
from akg import tvm,topi
from akg.utils import validation_check as vc_util
from akg.ops.math.div import div
from akg.ops.math.reciprocal import reciprocal
from akg.ops.math.mul import mul
from akg.utils import kernel_exec as utils

# define a string name of "float16"
FLOAT_16 = "float16"
# define a string name of "float32"
FLOAT_32 = "float32"
# define a string name of "int32"
INT_32 = "int32"
# define the PI
PI = 3.14159265
# define the expansion order of Tan series
TAN_EXPANSION_ORDER = 5
# define the number of times using the tan2x formula
TAN_2X_TIMES = 2


def get_attrs():
    """Get default attrs for tan."""
    default_attr_map = {
        "enable_auto_inline": False,
    }
    return default_attr_map

def _tan_expand(input_x):
    """calculating tan x = x + x^3/3 + 2*x^5/15 + 17*x^7/315 + 62*x^9/2835 + 1382*x^11/155925...(|x|<pi/2)"""
    # Taylor expansion coefficient
    factors = [1/3, 2/15, 17/315, 62/2835, 1382/155925]
    input_x_power = topi.multiply(input_x, input_x)
    iter_value = input_x
    res = input_x
    for i in range(TAN_EXPANSION_ORDER):
        if input_x.dtype == FLOAT_16 and utils.product_is_mini():
            iter_value = topi.multiply(input_x_power, iter_value)
            res = topi.add(res, topi.multiply(iter_value, tvm.const(factors[i], FLOAT_16)))
        else:
            iter_value = topi.multiply(input_x_power, iter_value)
            res = topi.add(res, topi.multiply(iter_value, factors[i]))
    return res

def _tan_2x_multi(input_x, times):
    """calculating tan x by calculating tan (x/2^times) and using double angle formula multiple times"""
    # Calculate tan (x/2^times)
    if input_x.dtype == FLOAT_16 and utils.product_is_mini():
        input_x_divide = topi.multiply(input_x, tvm.const(1.0/(2.0**times), FLOAT_16))
        res = _tan_expand(input_x_divide)
    else:
        input_x_divide = topi.multiply(input_x, 1.0/(2.0**times))
        res = _tan_expand(input_x_divide)
    while times != 0:
        # using double angle formula: tan 2x = 2*tan x/(1-tan x*tan x)
        if input_x.dtype == FLOAT_16 and utils.product_is_mini():
            res_numerator = topi.multiply(res, tvm.const(2.0, FLOAT_16))
            tanx_square = topi.multiply(res, res)
            res_denominator = topi.add(topi.multiply(tanx_square, tvm.const(-1.0, FLOAT_16)), tvm.const(1.0, FLOAT_16))
        else:
            res_numerator = topi.multiply(res, 2.0)
            tanx_square = topi.multiply(res, res)
            res_denominator = topi.add(topi.multiply(tanx_square, -1.0), 1.0)

        if utils.product_is_mini():
            res = mul(res_numerator, reciprocal(res_denominator))
        else:
            res = div(res_numerator, res_denominator)
        times = times - 1
    return res


def tan_compute(input_x):
    """tan compute implemention"""
    dtype = input_x.dtype

    # cast to type float32 when type is float16 in cloud and mini, or int32 in cloud
    if dtype == FLOAT_16 or dtype == FLOAT_32 or (dtype == INT_32 and not utils.product_is_mini()):
        input_x = topi.cast(input_x, FLOAT_32)
        # adjust x to [-pi/2,pi/2] using x = x-round(x/pi)*pi
        round_pi_div = akg.lang.cce.round(topi.multiply(input_x, tvm.const(1.0/PI, FLOAT_32)))
        round_pi_div = akg.lang.cce.cast_to(round_pi_div, FLOAT_32)
        input_x = topi.subtract(input_x, topi.multiply(round_pi_div, tvm.const(PI, FLOAT_32)))
    # cast to type float16 when type is int32 in mini
    elif dtype == INT_32 and utils.product_is_mini():
        input_x = topi.cast(input_x, FLOAT_16)
        # adjust x to [-pi/2,pi/2] using x = x-round(x/pi)*pi
        round_pi_div = akg.lang.cce.round(topi.multiply(input_x, tvm.const(1.0/PI, FLOAT_16)))
        round_pi_div = akg.lang.cce.cast_to(round_pi_div, FLOAT_16)
        input_x = topi.subtract(input_x, topi.multiply(round_pi_div, tvm.const(PI, FLOAT_16)))

    res = _tan_2x_multi(input_x, TAN_2X_TIMES)
    # cast the dtype to original dtype
    res = topi.cast(res, dtype)
    return res

@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def tan(input_x):
    """
    Calculate tan x = x + x^3/3 + 2*x^5/5 + 17*x^7/315 + 62*x^9/2835 + 1382*x^11/155925...(|x|<pi/2).

    Args:
        input_x (tvm.tensor.Tensor): Tensor of float32, float16, int32.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as input_x.
    """
    shape_input = input_x.shape
    dtype_input = input_x.dtype
    vc_util.check_shape(shape_input)
    vc_util.ops_dtype_check(dtype_input, [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT32])
    attrs = get_attrs()
    res = tan_compute(input_x)
    return res, attrs
