# Copyright 2019 Huawei Technologies Co., Ltd
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
from akg.ops.math.div import div
from akg.ops.math.cast import cast
from akg.ops.math.floor import floor
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import produce_shapes


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
            return akg.topi.subtract(a, akg.topi.multiply(b, cast(floor(div(a, b)), b.dtype)))

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

    _, _, out_shape = produce_shapes(vc_util.get_shape(x), vc_util.get_shape(y))
    x = akg.topi.broadcast_to(x, out_shape)
    y = akg.topi.broadcast_to(y, out_shape)

    # Scenarios for correcting calculation results are complex,
    # using absolute values can simplify the scenario:
    # truncatemod(x,y) = sign(x) * truncatemod(|x|, |y|)
    mod_abs = truncatemod_positive(akg.topi.abs(x), akg.topi.abs(y))
    mod = akg.topi.multiply(akg.topi.sign(x), mod_abs)
    return mod


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def truncatemod(x, y):
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

    vc_util.check_shape(x)
    vc_util.check_shape(y)
    vc_util.elemwise_dtype_check(x.dtype, y.dtype)
    dtype = x.dtype
    support_dtype = [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT32,
                     vc_util.DtypeForDavinci.INT8, vc_util.DtypeForDavinci.UINT8]
    if utils.product_is_mini():
        support_dtype = [vc_util.DtypeForDavinci.FLOAT16]

    vc_util.ops_dtype_check(dtype, support_dtype)

    if not utils.product_is_mini():
        # The high precision compute is required.
        # For brevity, lex x = 132.05, y = 131.95; x and y are very close, but the difference between trunc(x)=132
        # and trunc(y)=131 is 1
        if dtype != "float32":
            x = cast(x, "float32")
            y = cast(y, "float32")
        res = akg.topi.mod(x, y)
    else:
        res = _truncatemod_compute_mini(x, y)

    if res.dtype != dtype:
        res = cast(res, dtype)
    return res
