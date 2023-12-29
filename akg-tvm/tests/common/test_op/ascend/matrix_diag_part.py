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

"""operator dsl function: matrix_diag_part"""
import akg
from akg import tvm, topi
import akg.utils as utils
from akg.utils.format_transform import get_shape
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini

def matrix_diag_part_compute(input_diagonal, input_help):
    """matrix_diag_part compute implemention"""
    shape_input_diagonal = get_shape(input_diagonal)
    dtype_input_diagonal = input_diagonal.dtype
    if dtype_input_diagonal == "int8" or dtype_input_diagonal == "uint8":
        input_diagonal = topi.cast(input_diagonal, "float16")
        input_help = topi.cast(input_help, "float16")
    if dtype_input_diagonal == "int32" and product_is_mini():
        input_diagonal = topi.cast(input_diagonal, "float16")
        input_help = topi.cast(input_help, "float16")
        input_diagonal = topi.cast(input_diagonal, "float32")
        input_help = topi.cast(input_help, "float32")
    if dtype_input_diagonal == "int32" and not product_is_mini():
        input_diagonal = topi.cast(input_diagonal, "float32")
        input_help = topi.cast(input_help, "float32")
    res_vmul = topi.multiply(input_help, input_diagonal)

    if shape_input_diagonal[-2] < shape_input_diagonal[-1]:
        res = topi.sum(res_vmul, -1)
    else:
        res = topi.sum(res_vmul, -2)

    if dtype_input_diagonal == "int32" and product_is_mini():
        res = topi.cast(res, "float16")

    res = topi.cast(res, dtype_input_diagonal)
    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def matrix_diag_part(input_diagonal, input_help):
    """
    Calculate the batched diagonal part of a batched tensor.
    Note:
        input_help is a tensor with a diagonal element of 1 and other positions of 0,
        the last two dimensions can be unequal.

    Args:
        input_diagonal (tvm.tensor.Tensor): Tensor of float32, float16, int32, int8, uint8. The last two dimensions
                                            can be unequal.
        input_help (tvm.tensor.Tensor): Tensor of float32, float16, int32, int8, uint8, and with a diagonal element of 1
                                        and other positions of 0.
    Returns:
        tvm.tensor.Tensor, has the same type as input_diagonal, the shape dims is equal to dims(input_diagonal) - 1.
    """
    dtype_input_diagonal = input_diagonal.dtype
    dtype_input_help = input_help.dtype

    utils.elemwise_shape_check(input_help.shape, input_diagonal.shape)

    if len(input_help.shape) < 2:
        raise ValueError("Input tensors of rank>=2 are supported!")

    utils.ops_dtype_check([dtype_input_diagonal, dtype_input_help], [utils.DtypeForDavinci.ALL_FLOAT,
                                                                       utils.DtypeForDavinci.INT8,
                                                                       utils.DtypeForDavinci.INT32,
                                                                       utils.DtypeForDavinci.UINT8])
    res = matrix_diag_part_compute(input_diagonal, input_help)
    return res

