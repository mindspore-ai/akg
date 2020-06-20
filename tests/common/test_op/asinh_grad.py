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

"""operator dsl function: asinh_grad"""

from akg import tvm
from akg import topi
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils
from test_op.cosh import cosh_compute as cosh


@vc_util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor)
def asinh_grad(y, dy):
    """
    Gradient for asinh.

    Note:
        dx = dy * 1/cosh(y)

    Args:
        y (tvm.tensor.Tensor): tensor of type float16, float32.
        dy (tvm.tensor.Tensor): same type and shape as y.

    Returns:
        tvm.tensor.Tensor, same type and shape as y.
    """

    # mini product just used to infer
    if utils.product_is_mini():
        raise RuntimeError("The mini product does not support the asinh_grad operator")

    dtype = y.dtype
    vc_util.ops_dtype_check(y.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    vc_util.elemwise_dtype_check(dtype, dy.dtype)
    vc_util.check_shape(y.shape)
    vc_util.elemwise_shape_check(y.shape, dy.shape)

    if dtype == "float16":
        y = topi.cast(y, "float32")
        dy = topi.cast(dy, "float32")

    # dx = dy/cosh(y)
    dx = topi.divide(dy, cosh(y))

    if dx.dtype != dtype:
        dx = topi.cast(dx, dtype)

    return dx
