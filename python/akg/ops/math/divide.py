# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: divide"""
import akg.tvm
import akg.topi
import akg.utils as utils
from akg.ops.math.ascend.floor import floor
from akg.utils.dsl_create import produce_shapes
from akg.utils.kernel_exec import product_is_mini
from akg.ops.math.reciprocal import reciprocal
from .cast import cast


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def _div_ascend(data1, data2):
    """
    Calculates x/y, and returns an integer when inputs are all integers.

    When both arguments are integers, use integer division (also known as "floor division").
    When arguments are float numbers, use normal floating point division

    Note:
        div supports broadcasting.

    Args:
        data1 (tvm.tensor.Tensor): Tensor of type float16, float32, int32, int8 and uint8.
        data2 (tvm.tensor.Tensor): Tensor of type float16, float32, int32, int8 and uint8.

    Returns:
        tvm.tensor.Tensor, has the same type as data1 and data2.
    """

    utils.ops_dtype_check([data1.dtype, data2.dtype], utils.DtypeForDavinci.ALL_TYPES)
    utils.elemwise_dtype_check(data1.dtype, data2.dtype)
    dtype = data1.dtype

    shape1 = [x.value for x in data1.shape]
    shape2 = [x.value for x in data2.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)

    utils.auto_broadcast_check(shape1, shape2)
    n_shape1, n_shape2, out_shape = produce_shapes(shape1, shape2)
    if n_shape1 != out_shape:
        input1_cast = akg.topi.broadcast_to(data1, out_shape)
    else:
        input1_cast = data1
    if n_shape2 != out_shape:
        input2_cast = akg.topi.broadcast_to(data2, out_shape)
    else:
        input2_cast = data2

    if dtype in ("int32", "int8", "uint8"):
        input1p = cast(input1_cast, "float16", utils.CCE)
        input2p = cast(input2_cast, "float16", utils.CCE)
    else:
        input1p = input1_cast
        input2p = input2_cast


    if product_is_mini():
        input2p_rec = reciprocal(input2p, target=utils.CCE)
        res = akg.topi.multiply(input1p, input2p_rec)
    else:
        res = akg.topi.divide(input1p, input2p)

    if dtype in ("int8", "uint8"):
        res = floor(res, utils.CCE)
        res = cast(res, "float16", utils.CCE)
    if dtype in ("int32", "int8", "uint8"):
        res = cast(res, dtype, utils.CCE)

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def divide(lhs, rhs, target=utils.CCE):
    """
    Calculate divide.

    Args:
        lhs: The left tensor.
        rhs: The right tensor.

    Returns:
        tvm.tensor.Tensor.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return _div_ascend(lhs, rhs)
    shape_l = [x.value for x in lhs.shape]
    shape_r = [x.value for x in rhs.shape]
    utils.check_shape(shape_l)
    utils.check_shape(shape_r)
    utils.auto_broadcast_check(shape_l, shape_r)
    utils.elemwise_dtype_check(lhs.dtype, rhs.dtype)
    output = akg.topi.divide(lhs, rhs)

    return output
