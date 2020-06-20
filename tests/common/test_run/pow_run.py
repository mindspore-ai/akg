# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np
from akg.utils import kernel_exec as utils
from akg.utils import dsl_create as dsl_utils
from test_op import pow
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian

def pow_execute(shape1, shape2, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = pow_compile(shape1, shape2, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input1, input2, output = gen_data(dtype, shape1, shape2)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        mod = pow_compile(shape1, shape2, dtype, attrs)
        expect, input1, input2, output = gen_data(dtype, shape1, shape2)
        output = utils.mod_launch(mod, (input1, input2, output), expect=expect)
        rtol, atol = get_rtol_atol("pow", dtype)
        return (input1, input2), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape1, shape2):

    def pow_data_process(x, y):
        # For pow, if value of x is a negative number, the corresponding value of y must be an integer,
        # for example, the corresponding value of y is 1.0, -2.0, 3.0.
        x_b = np.broadcast_to(x, np.broadcast(x, y).shape)
        x_b_neg_index = np.where(x_b<0)
        if len(x_b_neg_index) > 0 and len(x_b_neg_index[0]) > 0:
            shape_len_diff = len(x_b.shape) - len(y.shape)
            y_int_index = list(x_b_neg_index[shape_len_diff:])
            for dim in range(len(y.shape)):
                if y.shape[dim] != x_b.shape[dim+shape_len_diff]:
                    if y.shape[dim] != 1:
                        raise ValueError("broadcast dismatch %s vs %s" %(y.shape[dim], x_b.shape[dim]))
                    y_int_index[dim] = np.array([0] * len(y_int_index[dim]))
            y_int_index = tuple(y_int_index)    
            y_int = y.astype(np.int32).astype(y.dtype)
            y[y_int_index] = y_int[y_int_index]
      
    input1 = random_gaussian(shape1, miu=1, sigma=0.1).astype(dtype)
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(dtype)
    pow_data_process(input1, input2)
    expect = np.power(input1, input2)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input1, input2, output


def pow_compile(shape1, shape2, dtype, attrs, kernel_name="pow", tuning=False):
    return utils.op_build_test(pow.pow_value, [shape1, shape2], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
