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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import transpose
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def transpose_data(shape, axes, dtype):
    input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    bench_mark = input.transpose(axes)
    return input, bench_mark


def transpose_execute(shape, axes, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = transpose_compile(shape, axes, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            bench_mark, input, output = gen_data(axes, dtype, shape)
            return mod, bench_mark, (input, output)
        else:
            return mod
    else:
        mod = transpose_compile(shape, axes, dtype, attrs)
        bench_mark, input, output = gen_data(axes, dtype, shape)
        output = utils.mod_launch(mod, (input, output), expect=bench_mark)

        # compare result
        rtol, atol = get_rtol_atol("transpose", dtype)
        compare_result = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=True)
        return input, output, bench_mark, compare_result


def gen_data(axes, dtype, shape):
    # Generate data
    input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    bench_mark = input.transpose(axes)
    # mod launch
    out_shape = ()
    for i in axes:
        out_shape = out_shape + (shape[i],)
    output = np.full(out_shape, np.nan, dtype)
    return bench_mark, input, output


def transpose_compile(shape, axes, dtype, attrs, kernel_name='transpose', tuning=False):
    op_attrs = [axes]
    return utils.op_build_test(transpose.transpose, [shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=tuning)