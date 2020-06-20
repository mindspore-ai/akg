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
from test_op.broadcast_to import broadcast_to
from gen_random import random_gaussian
from base import get_rtol_atol
from tensorio import compare_tensor


def broadcast_to_run(x_shape, x_dtype, shape, attrs):
    shapes = [x_shape]
    dtypes = [x_dtype]
    op_attrs = [shape]
    op_name = "broadcast_to"
    mod = utils.op_build_test(broadcast_to, shapes, dtypes, op_attrs=op_attrs,
                              kernel_name=op_name, attrs=attrs)
    bench_mark, inputs, output = gen_data(dtypes, shapes, shape)
    output = utils.mod_launch(mod, inputs + [output], expect=bench_mark)
    rtol, atol = get_rtol_atol(op_name, x_dtype)
    compare_res = compare_tensor(output, bench_mark, rtol=rtol, atol=atol)
    return inputs, output, bench_mark, compare_res


def gen_data(dtypes, shapes, shape):
    inputs = []
    for dtype, shape_ in zip(dtypes, shapes):
        input = random_gaussian(shape_).astype(dtype)
        inputs.append(input)
    expect = np.broadcast_to(inputs[0], shape)
    output = np.full(expect.shape, np.nan, dtypes[0])
    return expect, inputs, output
