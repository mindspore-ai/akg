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

"""operator dsl function: matrix_set_diag"""
import akg.lang.ascend
from akg import topi
import akg.utils as utils
from akg.utils.format_transform import get_shape
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini

def matrix_set_diag_compute(input_matrix, input_diagonal, input_help):
    """matrix_set_diag compute implemention"""
    shape_input = get_shape(input_matrix)
    input_dtype = input_matrix.dtype

    if input_dtype == "int8" or input_dtype == "uint8":
        input_matrix = topi.cast(input_matrix, "float16")
        input_diagonal = topi.cast(input_diagonal, "float16")
        input_help = topi.cast(input_help, "float16")
    if input_dtype == "int32" and product_is_mini():
        input_matrix = topi.cast(input_matrix, "float16")
        input_diagonal = topi.cast(input_diagonal, "float16")
        input_help = topi.cast(input_help, "float16")
        input_matrix = topi.cast(input_matrix, "float32")
        input_diagonal = topi.cast(input_diagonal, "float32")
        input_help = topi.cast(input_help, "float32")
    if input_dtype == "int32" and not product_is_mini():
        input_matrix = topi.cast(input_matrix, "float32")
        input_diagonal = topi.cast(input_diagonal, "float32")
        input_help = topi.cast(input_help, "float32")
    diag_tmp = topi.broadcast_to(input_diagonal, shape_input)
    help_tmp = topi.add(input_help, -1)
    help_y = topi.abs(help_tmp)

    res_vmul_x = topi.multiply(input_matrix, help_y)
    res_vmul_y = topi.multiply(diag_tmp, input_help)
    res = topi.add(res_vmul_x, res_vmul_y)

    if input_dtype == "int32" and product_is_mini():
        res = topi.cast(res, "float16")

    res = topi.cast(res, input_dtype)

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def matrix_set_diag(input_matrix, input_diagonal, input_help):
    """
    Return a batched matrix tensor with new batched diagonal values.

    Args:
        input_matrix (tvm.tensor.Tensor): Tensor of float32, float16, int32, int8, uint8. The last two dimensions
                                          can be unequal.
        input_diagonal (tvm.tensor.Tensor): Tensor of float32, float16, int32, int8, uint8.The last shape need equal
                                            to min(input_matrix[-1], input_matrix[-2]).
        input_help (tvm.tensor.Tensor): Tensor of float32, float16, int32, int8, uint8,and with a diagonal element of 1
                                        and other positions of 0.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as input_matrix.
    """
    shape_input = get_shape(input_matrix)
    shape_diag = get_shape(input_diagonal)
    shape_help = get_shape(input_help)
    dtype = input_matrix.dtype

    utils.check_shape(shape_input)
    utils.check_shape(shape_diag)
    utils.check_shape(shape_help)
    # Check help_matrix.
    if (len(shape_input) < 2) or (len(shape_help) < 2):
        raise RuntimeError("Only the rank of input tensors >= 2 are supported!")
    utils.elemwise_shape_check(shape_input, shape_help)

    # Check support dtype.
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT8,
                                    utils.DtypeForDavinci.INT32, utils.DtypeForDavinci.UINT8])

    # Adjust diag's shape according to input shape.
    # Extend the shape_diag dimension for broadcast.
    # if input_shape is [2,4,7,9] and shape_diag is [2,4,7] then new_shape is [2,4,7,1]
    # if input_shape is [2,4,9,7] and shape_diag is [2,4,7], then new_shape is [2,4,1,7]
    if shape_input[-2] <= shape_input[-1]:
        shape_b_newshape = list(shape_diag) + [1]
    # The penultimate dimension of the shape_diag is extended for broadcast.
    else:
        shape_b_newshape = list(shape_diag)
        shape_b_newshape.insert(-1, 1)
        
    input_diagonal = topi.reshape(input_diagonal, shape_b_newshape)
    res = matrix_set_diag_compute(input_matrix, input_diagonal, input_help)
    return res
