# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import two2fractal
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def two2fractal_execute(dim_size, format, dtype, attrs):
    # Generate data
    shape = dim_size
    support_formats = ['zN', 'zZ', 'nZ']
    assert format in support_formats
    assert len(shape) >= 2 and len(shape) <= 4

    # mod launch
    op_attrs = [format]

    default_attrs = { "polytops_parameter_shifting" : False, "enable_polytops" : "never" }
    attrs["pragma_disable_whole_component"] = False
    attrs.update(default_attrs)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = two2fractal_compile(shape, dtype, op_attrs, attrs, t)
        if t:
            bench_mark, input, output = gen_data(dtype, format, shape)
            return mod, bench_mark, (input, output)
        else:
            return mod
    else:
        mod = two2fractal_compile(shape, dtype, op_attrs, attrs)
        source_code = mod.imported_modules[0].get_source()
        bench_mark, input, output = gen_data(dtype, format, shape)
        output = utils.mod_launch(mod, (input, output), expect=bench_mark)

        # compare result
        rtol, atol = get_rtol_atol("tile", dtype)
        compare_result = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=True)
        return input, output, bench_mark, compare_result


def two2fractal_compile(shape, dtype, op_attrs, attrs, tuning=False):
    return utils.op_build_test(two2fractal.two2fractal, [shape], [dtype], op_attrs, kernel_name='two2fractal', attrs=attrs, tuning=tuning)


def gen_data(dtype, format, shape):
    input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    m, n = shape[-2], shape[-1]
    m1, n1 = m // 16, n // 16
    m0, n0 = 16, 16
    need_pad = m % 16 != 0 or n % 16 != 0
    if need_pad:
        pad_m, pad_n = (m + 15) // 16 * 16, (n + 15) // 16 * 16
        pad_shape = [x for x in shape]
        pad_shape[-1] = pad_n
        pad_shape[-2] = pad_m
        pad_input = np.zeros(pad_shape).astype(dtype)
        if len(shape) == 2:
            pad_input[:m, :n] = input
        elif len(shape) == 3:
            pad_input[:, :m, :n] = input
        elif len(shape) == 4:
            pad_input[:, :, :m, :n] = input
        m1, n1 = pad_m // 16, pad_n // 16
        reshape_shape = shape[:-2] + [m1, m0, n1, n0]
        reshape_input = pad_input.reshape(reshape_shape)
    else:
        reshape_shape = shape[:-2] + [m1, m0, n1, n0]
        reshape_input = input.reshape(reshape_shape)
    if format == 'zN':
        transpose_axis = [2, 0, 1, 3]
        new_shape = [n1, m1, m0, n0]
    elif format == 'zZ':
        transpose_axis = [0, 2, 1, 3]
        new_shape = [m1, n1, m0, n0]
    elif format == 'nZ':
        transpose_axis = [0, 2, 3, 1]
        new_shape = [m1, n1, n0, m0]
    transpose_axis = [x + len(shape) - 2 for x in transpose_axis]
    transpose_axis = [x for x in range(len(shape) - 2)] + transpose_axis
    new_shape = shape[:-2] + new_shape
    bench_mark = reshape_input.transpose(transpose_axis).astype('float16')
    # fractal only support float16
    output = np.full(new_shape, np.nan, 'float16')
    return bench_mark, input, output
