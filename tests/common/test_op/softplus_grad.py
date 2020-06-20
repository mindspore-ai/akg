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

"""operator dsl function: softplus_grad"""
import akg
from akg import tvm
from akg.ops.math.div import div

from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util, kernel_exec as utils
from akg.utils.dsl_create import produce_shapes


# define a scalar, value = 1
SCALAR_ONE = 1


def softplus_grad_compute(input_gradients, input_features):
    """compute for calculations of softplus gradients"""
    shape_dy = get_shape(input_gradients)
    shape_x = get_shape(input_features)
    dtype = input_gradients.dtype

    if list(shape_dy) != list(shape_x):
        shape_dy, shape_x, shape_max = produce_shapes(shape_dy, shape_x)
        input_gradients = akg.lang.cce.broadcast(
            input_gradients, shape_max, dtype)
        input_features = akg.lang.cce.broadcast(
            input_features, shape_max, dtype)
    else:
        shape_max = shape_dy

    if dtype != "float32":
        input_gradients = akg.lang.cce.cast_to(input_gradients, "float32")
        input_features = akg.lang.cce.cast_to(
            input_features, "float16" if utils.product_is_mini() else "float32")

    data_exp_tmp = akg.lang.cce.vexp(input_features)
    data_add_tmp = akg.lang.cce.vadds(data_exp_tmp, SCALAR_ONE)
    data_div_tmp = div(data_exp_tmp, data_add_tmp)
    res_tmp = akg.lang.cce.vmul(input_gradients, data_div_tmp)

    if dtype == "float16":
        res = akg.lang.cce.cast_to(res_tmp, "float16")
    elif dtype == "int32" or dtype == "int8" or dtype == "uint8":
        data_zero = akg.lang.cce.broadcast(
            tvm.const(0, "float16"), shape_max, "float16")
        res_min = akg.lang.cce.vmin(res_tmp, data_zero)
        res_max = akg.lang.cce.vmax(res_tmp, data_zero)
        res_max_int = akg.lang.cce.floor(res_max)
        res_min_int = akg.lang.cce.ceil(res_min)
        res = akg.lang.cce.vadd(res_max_int, res_min_int)
    else:
        res = res_tmp

    if dtype == "int8":
        res = akg.lang.cce.cast_to(res, "int8")
    elif dtype == "uint8":
        res = akg.lang.cce.cast_to(res, "uint8")

    return res


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def softplus_grad(data_dy, data_x):
    """
    Computes softplus gradients for a softplus operation.
    
    .. math::
        dx = \\dfrac{dy * e^x}{1 + e^x}

    Notes:
        Some value of result will be one less while dtype is "uint8".

    Args:
        data_dy (tvm.tensor.Tensor): The backpropagated gradients to
                                     the corresponding softplus operation.
        data_x (tvm.tensor.Tensor): The input_features passed as input
                                    to the corresponding softplus operation.
                                    source data type support "float16",
                                    "float32", "int32", "int8", "uint8".

    Returns:
        tvm.tensor.Tensor as gradients of data_x.
    """
    shape_dy = get_shape(data_dy)
    dtype_dy = data_dy.dtype
    shape_x = get_shape(data_x)
    dtype_x = data_x.dtype

    if dtype_dy != dtype_x:
        raise RuntimeError(
            "type of dy and type of x must be same, \
             while the types are different")
    else:
        dtype = dtype_dy

    vc_util.check_shape(shape_dy)
    vc_util.check_shape(shape_x)

    vc_util.ops_dtype_check(
        dtype,
        (vc_util.DtypeForDavinci.FLOAT16,
         vc_util.DtypeForDavinci.FLOAT32,
         vc_util.DtypeForDavinci.INT32,
         vc_util.DtypeForDavinci.INT8,
         vc_util.DtypeForDavinci.UINT8
         ) if not utils.product_is_mini() else \
        (vc_util.DtypeForDavinci.FLOAT16,
         vc_util.DtypeForDavinci.FLOAT32))

    return softplus_grad_compute(data_dy, data_x)
