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
from test_op import triangle
from akg.utils import kernel_exec as utils
from gen_random import random_gaussian

def triangle_execute(shape, const_value, lower, dtype, attrs):
    support_type = ['float16', 'float32']
    assert dtype in support_type
    assert len(shape) <= 2
    if attrs is None:
        attrs = {'enable_pre_poly_loop_partition': False}

    attrs['enable_pre_poly_loop_partition'] = False
    attrs['enable_post_poly_loop_partition'] = False
    attrs['enable_convert_if'] = True
    attrs['enable_double_buffer'] = False

    output_shape = shape
    if len(shape) == 1:
        output_shape = [shape[0], shape[0]]

    input, bench_mark = gen_data(shape, output_shape, const_value, lower, dtype)

    op_attrs = [const_value, lower]
    mod = triangle_compile(shape, dtype, op_attrs, attrs)
    source_code = mod.imported_modules[0].get_source()

    output = np.full(output_shape, np.nan, dtype)
    output = utils.mod_launch(mod, (input, output), expect=bench_mark)

    # compare result
    compare_result = compare_tensor(output, bench_mark, rtol=5e-3, equal_nan=True)
    return input, output, bench_mark, compare_result


def triangle_compile(shape, dtype, op_attrs, attrs):
    return utils.op_build_test(triangle.triangle, [shape], [dtype], op_attrs, kernel_name='triangle', attrs=attrs)


def gen_data(shape, output_shape, const_value, lower, dtype):
    input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    if len(shape) == 2:
        bench_mark = input
    else:
        bench_mark = np.zeros(output_shape).astype(dtype)
        for i in range(output_shape[0]):
            bench_mark[i] = input

    if lower:
        for i in range(output_shape[0]):
            bench_mark[i][i + 1:] = const_value
    else:
        for i in range(output_shape[0]):
            bench_mark[i][:i] = const_value

    return input, bench_mark
