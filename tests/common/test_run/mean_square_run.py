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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import mean_square
from akg.utils.dsl_create import get_reduce_out_shape
from gen_random import random_gaussian

def mean_square_run(shape, reduce_axis, keepdims, dtype, attrs):
    op_attrs = [reduce_axis, keepdims]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(mean_square.mean_square, [shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input1, output = gen_data(dtype, keepdims, reduce_axis, shape)
            return mod, expect, (input1, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(mean_square.mean_square, [shape], [dtype], op_attrs, attrs=attrs)
        expect, input1, output = gen_data(dtype, keepdims, reduce_axis, shape)
        output = utils.mod_launch(mod, (input1, output), expect=expect)
        return input1, output, expect, compare_tensor(output, expect, rtol=5e-3, atol=5e-3, equal_nan=True)


def gen_data(dtype, keepdims, reduce_axis, shape):
    input1 = random_gaussian(shape, miu=1, sigma=0.1)
    input1 = input1.astype(dtype)
    input1_square = np.square(input1)
    expect = np.mean(input1_square, axis=reduce_axis, keepdims=keepdims)
    out_shape = get_reduce_out_shape(shape, axis=reduce_axis, keepdims=keepdims)
    output = np.full(out_shape, np.nan, dtype)
    return expect, input1, output
