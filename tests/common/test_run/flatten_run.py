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

import numpy as np

from akg.utils import kernel_exec as utils
from tests.common.test_op.flatten import flatten
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor

def flatten_run(x_shape, x_dtype, attrs):
    shapes = [x_shape]
    dtypes = [x_dtype]
    mod = utils.op_build_test(flatten, shapes, dtypes,
                              kernel_name="flatten", attrs=attrs)
    bench_mark, inputs, output = gen_data(x_dtype, x_shape)
    output = utils.mod_launch(mod, inputs + [output], expect=bench_mark)
    rtol, atol = get_rtol_atol("flatten", x_dtype)
    compare_res = compare_tensor(output, bench_mark, rtol=rtol, atol=atol)
    return inputs, output, bench_mark, compare_res


def gen_data(dtype, shape):
    inputs = np.random.uniform(low=1, high=255, size=shape).astype(dtype)
    size = 1
    for i in range(1, len(shape)):
        size = size * shape[i]
    expect = np.reshape(inputs, [shape[0], size])
    output = np.full([shape[0], size], np.nan, dtype)
    return expect, [inputs], output
