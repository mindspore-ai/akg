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

"""selu_run"""
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import selu
from tests.common.base import get_rtol_atol

# define selu oprator's required constants
ALPHA = 1.67326324235
SCALE = 1.05070098736
# define product of scale and alpha
SCALE_ALPHA_PRODUCT = 1.75809934085
# define a scalar, value = -1, the calculation of exp need minus one
SCALAR_NEGATIVE_ONE = -1

def selu_run(shape, dtype, attrs):
    """selu_run implementation"""
    mod = utils.op_build_test(selu.selu, [shape], [dtype], kernel_name='selu', op_attrs=[], attrs=attrs)
    args, exp_output, input_data = gen_data(dtype, shape)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("selu", dtype)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return input_data, acu_output, exp_output, testcase_result


def gen_data(dtype, shape):
    # result_numpy
    if dtype == 'int8':
        low_bound = -128
        high_bound = 127
    elif dtype == 'int32':
        low_bound = -1000
        high_bound = 1000
    else:
        low_bound = -1.0
        high_bound = 1.0

    input_data = np.random.uniform(low=low_bound, high=high_bound, size=tuple(shape)).astype(dtype)
    if dtype in ("float16", "float32"):
        input_data = input_data.astype("float32")
    else:
        input_data = input_data.astype("float16")
    tensor_zero = np.multiply(input_data, 0)
    # generate negative_res and positive_res to compute
    # When the element value is greater than 0 and less than 0
    negative_res = np.minimum(input_data, tensor_zero)
    positive_res = np.maximum(input_data, tensor_zero)
    exp_res = np.exp(negative_res)
    sub_res = np.add(exp_res, SCALAR_NEGATIVE_ONE)
    negative_muls_res = np.multiply(sub_res, SCALE_ALPHA_PRODUCT)
    if dtype == "int8":
        negative_muls_res = np.ceil(negative_muls_res)

    positive_muls_res = np.multiply(positive_res, SCALE)
    exp_output = np.add(negative_muls_res, positive_muls_res)

    # cast to ori_dtype
    if dtype == "float16" or dtype == "int8" or dtype == "int32":
        exp_output = exp_output.astype(dtype)

    input_data = input_data.astype(dtype)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = [input_data, output]
    return args, exp_output, input_data
