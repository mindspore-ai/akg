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

"""operator dsl function: acosh"""
import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils

from tests.common.test_op.asinh import log_compute_mini_impl


def _sqrt_mini_vsqrt_newton_iter(x):
    """sqrt compute on mini with the Newton's Iteration of vrsqrt"""
    def vsqrt_newton_iter(data):
        """compute vsqrt with newton_iter"""
        data_rsqrt = akg.topi.rsqrt(data)
        # vrsqrt newton_iter: x(n+1) = x(n)*(3-a*x(n)^2)/2
        steps = 3
        half = akg.tvm.const(0.5, x.dtype)
        shape = data.shape
        for i in range(steps):
            data_rsqrt = akg.tvm.compute(shape, lambda *indice: half * data_rsqrt(*indice) *
                                         (3 - data(*indice) * data_rsqrt(*indice) * data_rsqrt(*indice)),
                                         name="data_rsqrt_%s" % i)
        return data_rsqrt

    x_rsqrt = vsqrt_newton_iter(x)
    x_sqrt = akg.topi.multiply(x, x_rsqrt)
    return x_sqrt


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def acosh(x):
    r"""
    Compute acosh function.

    .. math:: acosh(x) = log(x+\sqrt{x*x-1})

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32. Each entry
        in it must be in `[1, inf)`

    Returns:
       tvm.tensor.Tensor, has the same type and shape as x.
    """
    # check shape
    vc_util.check_shape(x)

    # check input tensor data_type
    vc_util.ops_dtype_check(x.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    dtype = x.dtype

    if dtype == "float16":
        # To avoid overflow and higher accuracy, x is casted to float32
        x = akg.topi.cast(x, "float32")

    # acosh(x) = log(x + sqrt(x*x-1))
    x_square = akg.topi.multiply(x, x)
    x_square_sub = akg.topi.subtract(x_square, 1)

    if utils.product_is_mini():
        sqrt_value = _sqrt_mini_vsqrt_newton_iter(x_square_sub)
    else:
        sqrt_value = akg.topi.sqrt(x_square_sub)

    sqrt_add = akg.topi.add(sqrt_value, x)

    if utils.product_is_mini():
        res = log_compute_mini_impl(sqrt_add)
    else:
        res = akg.topi.log(sqrt_add)

    if res.dtype != dtype:
        res = akg.topi.cast(res, dtype)

    if utils.product_is_mini():
        attrs = {"enable_auto_inline": False}
        return res, attrs
    return res
