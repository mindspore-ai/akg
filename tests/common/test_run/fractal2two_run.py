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
from test_op import fractal2two
from gen_random import random_gaussian

def fractal2two_execute(dim_size, shape_origin, format, dtype, out_dtype, attrs):
    supportFracalList = ['zN', 'zZ']
    # Generate data
    shape = dim_size

    assert format in supportFracalList
    assert len(shape) >= 4
    op_attrs = [out_dtype, shape_origin, format]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = fractal2two_compile(shape, dtype, op_attrs, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            output, input, bench_mark = gen_data(shape, dtype, shape_origin, format, out_dtype)
            return mod, bench_mark, (input, output)
        else:
            return mod
    else:
        # mod launch
        mod = fractal2two_compile(shape, dtype, op_attrs, attrs)
        source_code = mod.imported_modules[0].get_source()
        output, input, bench_mark = gen_data(shape, dtype, shape_origin, format, out_dtype)
        output = utils.mod_launch(mod, (input, output), expect=bench_mark)

        # compare result
        compare_result = compare_tensor(output, bench_mark, rtol=5e-3, equal_nan=True)
        return input, output, bench_mark, compare_result


def fractal2two_compile(shape, dtype, op_attrs, attrs, kernel_name='fractal2two', tuning=False):
    return utils.op_build_test(fractal2two.fractal2two, [shape], [dtype], op_attrs, kernel_name='fractal2two', attrs=attrs, tuning=tuning)


def gen_data(shape, dtype, shape_origin, format, out_dtype):
    if format == 'zN':
        input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
        n1, m1, m0, n0 = shape[-4:]
        new_shape = shape[:-4] + [m1 * m0, n1 * n0]
        tranpose_axis = [1, 2, 0, 3]
    elif format == 'zZ':
        input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
        m1, n1, m0, n0 = shape[-4:]
        new_shape = shape[:-4] + [m1 * m0, n1 * n0]
        tranpose_axis = [0, 2, 1, 3]

    tranpose_axis = [x + len(shape) - 4 for x in tranpose_axis]
    tranpose_axis = [i for i in range(len(shape) - 4)] + tranpose_axis
    bench_mark = input.transpose(tranpose_axis).reshape(new_shape)
    if new_shape != shape_origin:
        if len(shape_origin) == 2:
            bench_mark = bench_mark[:shape_origin[0], :shape_origin[1]].astype(out_dtype)
        elif len(shape_origin) == 3:
            bench_mark = bench_mark[:, shape_origin[0], :shape_origin[1]].astype(out_dtype)
        elif len(shape_origin) == 4:
            bench_mark = bench_mark[:, :, shape_origin[0], :shape_origin[1]].astype(out_dtype)
        new_shape = shape_origin
    output = np.full(new_shape, np.nan, out_dtype)
    return output, input, bench_mark
