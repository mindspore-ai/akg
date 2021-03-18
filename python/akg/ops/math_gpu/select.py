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

"""operator dsl function: select"""
import akg.topi
import akg.tvm
import akg.lang.cce
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.utils import kernel_exec as utils

VALUE_ONE = 1

def select_compute(condition, x1, x2):
    """select compute implementation"""
    shape = get_shape(x1)
    con_shape = get_shape(condition)
    num_dtype = x1.dtype
    bool_dtype = condition.dtype

    if num_dtype in ("int8", "uint8"):
        x1_dtype = "float32"
        ones = akg.lang.cce.broadcast(akg.tvm.const(VALUE_ONE, dtype="float32"),
                                      shape, output_dtype="float32")
        x1 = akg.topi.cast(x1, "float32")
        x2 = akg.topi.cast(x2, "float32")
    else:
        x1_dtype = num_dtype
        ones = akg.lang.cce.broadcast(akg.tvm.const(VALUE_ONE, dtype=num_dtype),
                                      shape, output_dtype=num_dtype)

    if bool_dtype == "int8":
        if x1_dtype == "int32":
            condition_dtype = akg.lang.cce.ceil(condition)
        else:
            condition_dtype = akg.topi.cast(condition, x1_dtype)
    else:
        if x1_dtype == "int32":
            condition_dtype = condition
        else:
            condition_dtype = akg.topi.cast(condition, x1_dtype)

    if list(con_shape) != list(shape):
        condition_dtype = akg.lang.cce.broadcast(condition_dtype, shape)

    vinsn_support_dtype = ("float16", "float32")
    if utils.product_is_mini():
        vinsn_support_dtype = ("float16", )
    if num_dtype in vinsn_support_dtype:
        res = akg.topi.where(condition_dtype, x1, x2)
    else:
        condition_opp = akg.lang.cce.vsub(ones, condition_dtype)
        temp_x = akg.lang.cce.vmul(x1, condition_dtype)
        temp_y = akg.lang.cce.vmul(x2, condition_opp)
        res = akg.lang.cce.vadd(temp_x, temp_y)
    if num_dtype in ("int8", "uint8"):
        res = akg.topi.cast(res, num_dtype)
    return res


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def select(condition, x1, x2):
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

    """
    shape_x1 = get_shape(x1)
    shape_x2 = get_shape(x2)
    con_shape = get_shape(condition)
    vc_util.elemwise_shape_check(shape_x1, shape_x2)
    vc_util.elemwise_dtype_check(x1.dtype, x2.dtype, [vc_util.DtypeForDavinci.ALL_FLOAT,
                                                      vc_util.DtypeForDavinci.INT8, vc_util.DtypeForDavinci.INT32,
                                                      vc_util.DtypeForDavinci.UINT8])
    vc_util.ops_dtype_check(condition.dtype, [vc_util.DtypeForDavinci.INT8, vc_util.DtypeForDavinci.INT32])
    vc_util.auto_broadcast_check(con_shape, shape_x1)
    res = select_compute(condition, x1, x2)
    return res
