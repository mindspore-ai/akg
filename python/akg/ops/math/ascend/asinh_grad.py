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

"""operator dsl function: asinh_grad"""

from akg import tvm
from akg import topi
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini
from .cosh import cosh_compute as cosh


@utils.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (str, type(None)))
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
    
    Supported Platforms:
        'Ascend'
    """

    # mini product just used to infer
    if product_is_mini():
        raise RuntimeError("The mini product does not support the asinh_grad operator")

    dtype = y.dtype
    utils.ops_dtype_check(y.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.elemwise_dtype_check(dtype, dy.dtype)
    utils.check_shape(y.shape)
    utils.elemwise_shape_check(y.shape, dy.shape)

    if dtype == "float16":
        y = topi.cast(y, "float32")
        dy = topi.cast(dy, "float32")

    dx = topi.divide(dy, cosh(y))

    if dx.dtype != dtype:
        dx = topi.cast(dx, dtype)

    return dx
