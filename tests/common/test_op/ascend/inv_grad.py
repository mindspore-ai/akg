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

"""operator dsl function: inv_grad"""

import akg
import akg.tvm
import akg.utils as utils

SCALAR_NEGATIVE_ONE = -1

def inv_grad_compute(input_y, input_dy):
    """inv_grad compute implementation"""
    shape_y = akg.lang.ascend.util.shape_to_list(input_y.shape)
    dtype = input_y.dtype

    inv_const = akg.tvm.const(SCALAR_NEGATIVE_ONE, dtype=dtype)
    if dtype in ("float16", "int8"):
        inv_const = akg.tvm.const(SCALAR_NEGATIVE_ONE, dtype="float32")
        input_y = akg.lang.ascend.cast_to(input_y, "float32")
        input_dy = akg.lang.ascend.cast_to(input_dy, "float32")
        const_res = akg.lang.ascend.vmuls(input_y, inv_const)
    elif dtype in ("int32",):
        inv_const = akg.lang.ascend.broadcast(inv_const, shape_y, "int32")
        const_res = akg.lang.ascend.vmul(inv_const, input_y)
    else:
        const_res = akg.lang.ascend.vmuls(input_y, inv_const)
    vmul_res = akg.lang.ascend.vmul(const_res, input_y)
    res = akg.lang.ascend.vmul(vmul_res, input_dy)

    if dtype in ("float16", "int8"):
        res = akg.lang.ascend.cast_to(res, dtype, f1628_int_flag=True)

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def inv_grad(input_y, input_dy):
    """
    Calculate data's Reciprocal grad,dx = -1 * input_dy * input_y * input_y.

    Args:
        input_y (tvm.tensor.Tensor): Tensor of type float16, float32, int8, int32.
        input_dy (tvm.tensor.Tensor): Tensor of type float16, float32, int8, int32.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as input_y.
    """
    # Check shapes and dtypes.
    utils.elemwise_shape_check(input_y.shape, input_dy.shape)
    utils.elemwise_dtype_check(input_y.dtype, input_dy.dtype, supported_type=["float16", "float32", "int8", "int32"])

    res = inv_grad_compute(input_y, input_dy)
    return res
