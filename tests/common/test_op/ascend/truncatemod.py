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

"""truncatemod"""

import akg.topi
import akg.tvm
from akg.ops.math import Divide, Cast
from akg.ops.math.ascend import Floor
import akg.utils as utils
from akg.utils.dsl_create import produce_shapes
from akg.utils.kernel_exec import product_is_mini

def _truncatemod_compute_mini(x, y):
    """
    Computes truncatemod value of x and y on mini device.
    Args:
        x(tvm.tensor.Tensor): Tensor, float16.
        y(tvm.tensor.Tensor): Tensor with same type as x.
    Returns:
        tvm.tensor.Tensor of same type as x.
    """

    def truncatemod_positive(x_abs, y_abs):
        """Computes truncatemod value for positive number"""
        x_abs_fp32 = akg.topi.cast(x_abs, "float32")
        y_abs_fp32 = akg.topi.cast(y_abs, "float32")

        def truncatemod_func(a, b):
            """function for truncatemod formula"""
            # For positive numbers, floor and trunc are equivalent
            return akg.topi.subtract(a, akg.topi.multiply(b, Cast(Floor(Divide(a, b, utils.CCE)), b.dtype, target=utils.CCE)))

        mod_value = truncatemod_func(x_abs_fp32, y_abs_fp32)

        # Because there are precision errors in division on mini, etc.,
        # the calculation results need to be corrected
        mod_value = truncatemod_func(mod_value, y_abs_fp32)
        mod_value = akg.topi.cast(mod_value, "float16")
        mod_value = akg.tvm.compute(mod_value.shape,
                                    lambda *indice: akg.tvm.expr.Select(mod_value(*indice) >= y_abs(*indice),
                                                                        mod_value(*indice) - y_abs(*indice),
                                                                        mod_value(*indice)),
                                    name="mod_value")
        return mod_value

    _, _, out_shape = produce_shapes(utils.get_shape(x), utils.get_shape(y))
    x = akg.topi.broadcast_to(x, out_shape)
    y = akg.topi.broadcast_to(y, out_shape)

    # Scenarios for correcting calculation results are complex,
    # using absolute values can simplify the scenario:
    # truncatemod(x,y) = Sign(x) * truncatemod(|x|, |y|)
    mod_abs = truncatemod_positive(akg.topi.abs(x), akg.topi.abs(y))
    mod = akg.topi.multiply(akg.topi.sign(x), mod_abs)
    return mod


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def truncatemod(x, y, target=utils.CCE):
    """
    Computes remainder of division(x / y).

    Note:
        res = x - y*trunc(x/y)

    Args:
        x(tvm.tensor.Tensor): Input tensor, support float16 on mini device, while support
                           int32, int8, uint8, float16, float32 on cloud ones.
        y(tvm.tensor.Tensor): Tensor with same type as input tensor x.

    Returns:
        tvm.tensor.Tensor of same type as input tensors.
    """

    utils.check_shape(x)
    utils.check_shape(y)
    utils.elemwise_dtype_check(x.dtype, y.dtype)
    dtype = x.dtype
    support_dtype = [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32,
                     utils.DtypeForDavinci.INT8, utils.DtypeForDavinci.UINT8]
    if product_is_mini():
        support_dtype = [utils.DtypeForDavinci.FLOAT16]

    utils.ops_dtype_check(dtype, support_dtype)

    if not product_is_mini():
        # The high precision compute is required.
        # For brevity, lex x = 132.05, y = 131.95; x and y are very close, but the difference between trunc(x)=132
        # and trunc(y)=131 is 1
        if dtype != "float32":
            x = Cast(x, "float32", target=target)
            y = Cast(y, "float32", target=target)
        res = akg.topi.mod(x, y)
    else:
        res = _truncatemod_compute_mini(x, y)

    if res.dtype != dtype:
        res = Cast(res, dtype, target=target)
    return res
