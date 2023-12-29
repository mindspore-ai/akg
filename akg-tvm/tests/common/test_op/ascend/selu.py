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

"""operator dsl function: selu"""
import akg.lang.ascend
from akg import tvm, topi
import akg.utils as utils
from akg.ops.math import exp


# define selu oprator's required constants
ALPHA = 1.67326324235
SCALE = 1.05070098736
# define product of scale and alpha
SCALE_ALPHA_PRODUCT = 1.75809934085
# define a scalar, value = -1, the calculation of exp need minus one
SCALAR_NEGATIVE_ONE = -1


def selu_compute(input_data):
    """selu compute implemention"""
    # if input_dtype is float16,convert it to float32
    dtype = input_data.dtype
    if dtype == "float16" or dtype == "float32":
        input_data = topi.cast(input_data, "float32")
        type_tmp = "float32"
    else:
        input_data = topi.cast(input_data, "float16")
        type_tmp = "float16"

    # generate tensor_zero to be compared
    tensor_zero = topi.multiply(input_data, tvm.const(0, dtype=type_tmp))
    # generate negative_res and positive_res to compute
    # When the element value is greater than 0 and less than 0
    negative_res = topi.minimum(input_data, tensor_zero)
    positive_res = topi.maximum(input_data, tensor_zero)
    exp_res = exp(negative_res, 'cce')
    sub_res = topi.add(exp_res, tvm.const(SCALAR_NEGATIVE_ONE, dtype=type_tmp))
    negative_muls_res = topi.multiply(sub_res, tvm.const(SCALE_ALPHA_PRODUCT, dtype=type_tmp))
    if dtype == "int8":
        negative_muls_res = akg.lang.ascend.ceil(negative_muls_res)

    positive_muls_res = topi.multiply(positive_res, tvm.const(SCALE, dtype=type_tmp))
    res = topi.add(negative_muls_res, positive_muls_res)
    # cast to ori_dtype
    if dtype == "float16" or dtype == "int8" or dtype == "int32":
        res = topi.cast(res, dtype)

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor)
def selu(input_data):
    """
    Computes scaled exponential linear.
        selu(x) = scale * alpha * (exp(input_data) - 1) if input_data < 0,
        selu(x) = scale * input_data                    otherwise.
    where alpha = 1.6732632423543772848170429916717, scale =  1.0507009873554804934193349852946.

    Args:
        input_data (tvm.tensor.Tensor): Tensor of type float16, float32, int8, int32.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as input_data.
   """
    # check dtype and shape
    utils.check_shape(input_data.shape)
    utils.ops_dtype_check(input_data.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.ALL_INT])

    res = selu_compute(input_data)
    return res
