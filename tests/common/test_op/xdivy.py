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

"""operator dsl function: xdivy"""
import akg
from akg import tvm
from akg.ops.math.div import div
from akg.utils.format_transform import get_shape
from akg.utils.dsl_create import produce_shapes
from akg.utils import validation_check as vc_util


# define a scalar , value = 1
SCALAR_ONE = 1
# minimun num of float32 2**(-126)
MININUM_NUM_FLOAT = 2**(-126)
# minimun num of float16 2**(-24)
MININUM_NUM_HALF = 2**(-24)
# max num of float32 is 2**126, but cce can only support 2**62,
# so use 62/62/2 to adaptor 149
MAX_ONE_CONST_FLOAT = 2**62
MAX_TWO_CONST_FLOAT = 2**2
# max num of float16 is 2**24, but cce can only support 2**12,
# so use 12/12 to adaptor 24
MAX_CONST_HALF = 2**12

def xdivy_compute(input_x, input_y):
    """xdivy compute"""
    _, _, shape_res = produce_shapes(get_shape(input_x), get_shape(input_y))
    vc_util.check_shape(shape_res)

    dtype = input_x.dtype

    broadcast_x = akg.lang.cce.broadcast(input_x, shape_res)
    broadcast_y = akg.lang.cce.broadcast(input_y, shape_res)
    broadcast_one = akg.lang.cce.broadcast(
        tvm.const(SCALAR_ONE, dtype), shape_res, dtype)

    abs_x = akg.lang.cce.vabs(broadcast_x)
    abs_y = akg.lang.cce.vabs(broadcast_y)
    add_x_y = akg.lang.cce.vadd(abs_x, abs_y)

    if dtype == "float32":
        data_min = akg.lang.cce.broadcast(
            tvm.const(MININUM_NUM_FLOAT, dtype=dtype), shape_res, dtype)
    elif dtype == "float16":
        data_min = akg.lang.cce.broadcast(
            tvm.const(MININUM_NUM_HALF, dtype=dtype), shape_res, dtype)

    zero_x_y = akg.lang.cce.vmin(add_x_y, data_min)

    if dtype == "float32":
        data_mul1 = akg.lang.cce.vmuls(
            zero_x_y, tvm.const(MAX_ONE_CONST_FLOAT, dtype=dtype))
        data_mul2 = akg.lang.cce.vmuls(
            data_mul1, tvm.const(MAX_ONE_CONST_FLOAT, dtype=dtype))
        mul_data = akg.lang.cce.vmuls(
            data_mul2, tvm.const(MAX_TWO_CONST_FLOAT, dtype=dtype))
    elif dtype == "float16":
        data_mul1 = akg.lang.cce.vmuls(
            zero_x_y, tvm.const(MAX_CONST_HALF, dtype=dtype))
        mul_data = akg.lang.cce.vmuls(
            data_mul1, tvm.const(MAX_CONST_HALF, dtype=dtype))

    sub_x_y_zero = akg.lang.cce.vsub(mul_data, broadcast_one)
    abs_x_y_zero = akg.lang.cce.vabs(sub_x_y_zero)
    input_y_revised = akg.lang.cce.vadd(broadcast_y, abs_x_y_zero)

    if dtype == "float16":
        broadcast_x = akg.lang.cce.cast_to(broadcast_x, "float32")
        input_y_revised = akg.lang.cce.cast_to(input_y_revised, "float32")

    res = div(broadcast_x, input_y_revised)

    if dtype == "float16":
        res = akg.lang.cce.cast_to(res, dtype)

    return res


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def xdivy(data_x1, data_x2):
    """
    Calculate data_x1 divided by data_x2.

    .. math::
        y = \\left\\{
	    \\begin{aligned}
		0, && if \\quad x1 == 0 \\\\
		\\dfrac{x1}{x2}, && otherwise
	    \\end{aligned}
	\\right.

    Args:
        data_x1 (tvm.tensor.Tensor): Tensor of dtype "float16" or "float32"
        data_x2 (tvm.tensor.Tensor): Tensor of dtype "float16" or "float32"

    Returns:
        tvm.tensor.Tensor
    """
    shape_x1 = get_shape(data_x1)
    shape_x2 = get_shape(data_x2)
    
    vc_util.check_shape(shape_x1)
    vc_util.check_shape(shape_x2)

    vc_util.elemwise_dtype_check(data_x1.dtype, data_x2.dtype)
    dtype = data_x1.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    return xdivy_compute(data_x1, data_x2)
