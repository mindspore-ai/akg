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

"""operator dsl function: select"""
import akg.topi
import akg.tvm
import akg.lang.ascend
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.utils.kernel_exec import product_is_mini


def select_compute(condition, x1, x2, target=utils.CCE):
    """select compute implementation"""
    shape = get_shape(x1)
    con_shape = get_shape(condition)
    num_dtype = x1.dtype
    bool_dtype = condition.dtype
    cast_op = akg.lang.ascend.cast_to if target == utils.CCE else akg.topi.cast
    if num_dtype in ("int8", "uint8"):
        x1_dtype = "float32"
        ones = akg.lang.ascend.broadcast(akg.tvm.const(1, dtype="float32"),
                                      shape, output_dtype="float32")
        x1 = cast_op(x1, "float32")
        x2 = cast_op(x2, "float32")
    else:
        x1_dtype = num_dtype
        ones = akg.lang.ascend.broadcast(akg.tvm.const(1, dtype=num_dtype),
                                      shape, output_dtype=num_dtype)

    if bool_dtype == "int8":
        if x1_dtype == "int32":
            condition_dtype = akg.lang.ascend.ceil(condition)
        else:
            condition_dtype = cast_op(condition, x1_dtype)
    else:
        if x1_dtype == "int32":
            condition_dtype = condition
        else:
            condition_dtype = cast_op(condition, x1_dtype)

    if list(con_shape) != list(shape):
        condition_dtype = akg.lang.ascend.broadcast(condition_dtype, shape)

    vinsn_support_dtype = ("float16", "float32")
    if product_is_mini():
        vinsn_support_dtype = ("float16", )
    if num_dtype in vinsn_support_dtype:
        res = akg.topi.where(condition_dtype, x1, x2)
    else:
        # For data types that are not supported by the vector instruction (vcmp and vsel),
        # if the `topi.where` is directly used, the related instructions generated in the .cce file
        # are scalar instructions such as `cond ? x1 : x2`, which is very inefficient.
        # Therefore, other equivalent calculation methods are adopted.
        condition_opp = akg.lang.ascend.vsub(ones, condition_dtype)
        temp_x = akg.lang.ascend.vmul(x1, condition_dtype)
        temp_y = akg.lang.ascend.vmul(x2, condition_opp)
        res = akg.lang.ascend.vadd(temp_x, temp_y)
    if num_dtype in ("int8", "uint8"):
        res = cast_op(res, num_dtype)
    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def select(condition, x1, x2, target=utils.CCE):
    """
    Selects elements from x1 or x2, depending on condition.
    Note:
        every parmas' shape need legal, can support condition's shape broadcast.

    Args:
        condition (tvm.tensor.Tensor): Tensor of type int8, int32, must be 0 or 1.
        x1 (tvm.tensor.Tensor): Tensor of type float16, float32, int8, int32, uint8.
        x2 (tvm.tensor.Tensor): Tensor of type float16, float32, int8, int32, uint8.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as x1.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    shape_x1 = get_shape(x1)
    shape_x2 = get_shape(x2)
    con_shape = get_shape(condition)
    utils.elemwise_shape_check(shape_x1, shape_x2)
    utils.elemwise_dtype_check(x1.dtype, x2.dtype, [utils.DtypeForDavinci.ALL_FLOAT,
                                                      utils.DtypeForDavinci.INT8, utils.DtypeForDavinci.INT32,
                                                      utils.DtypeForDavinci.UINT8])
    utils.ops_dtype_check(condition.dtype, [utils.DtypeForDavinci.INT8, utils.DtypeForDavinci.INT32])
    utils.auto_broadcast_check(con_shape, shape_x1)
    res = select_compute(condition, x1, x2, target)
    return res
