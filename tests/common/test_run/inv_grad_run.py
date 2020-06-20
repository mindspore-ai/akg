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

"""inv_grad_run"""
import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import inv_grad
from base import get_rtol_atol


def inv_grad_run(shape, dtype, attrs):
    """inv_grad_run implementation"""
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(inv_grad.inv_grad, [shape, shape], [dtype, dtype], kernel_name=kernel_name,
                                  op_attrs=[], attrs=attrs, tuning=t)
        if t:
            args, exp_output, input_y, inputs_dy = gen_data(dtype, shape)
            return mod, exp_output, args
        else:
            return mod
    else:
        mod = utils.op_build_test(inv_grad.inv_grad, [shape, shape], [dtype, dtype], kernel_name='inv_grad',
                                  op_attrs=[], attrs=attrs)
        args, exp_output, input_y, input_dy = gen_data(dtype, shape)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)
        # compare result
        rtol, atol = get_rtol_atol("inv_grad", dtype)
        testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

        return [input_y, input_dy], acu_output, exp_output, testcase_result


def gen_data(dtype, shape):
    # generate data
    if dtype == 'int8':
        low_bound = -128
        high_bound = 127
    elif dtype == 'int32':
        low_bound = -1000
        high_bound = 1000
    else:
        low_bound = -1.0
        high_bound = 1.0

    input_y = np.random.uniform(low=low_bound, high=high_bound, size=tuple(shape)).astype(dtype)
    input_dy = np.random.uniform(low=low_bound, high=high_bound, size=tuple(shape)).astype(dtype)
    if dtype in ('float16', 'int8'):
        input_y = input_y.astype("float32")
        input_dy = input_dy.astype("float32")
    exp_output = -1 * np.multiply(np.multiply(input_y, input_y), input_dy)
    if dtype in ('float16'):
        exp_output = exp_output.astype(dtype)
        input_y = input_y.astype(dtype)
        input_dy = input_dy.astype(dtype)
    elif dtype in ('int8'):
        exp_output[exp_output > 127] = 127.0
        exp_output[exp_output < -128] = -128.0
        exp_output = exp_output.astype(dtype)
        input_y = input_y.astype(dtype)
        input_dy = input_dy.astype(dtype)

    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = [input_y, input_dy, output]
    return args, exp_output, input_y, input_dy
