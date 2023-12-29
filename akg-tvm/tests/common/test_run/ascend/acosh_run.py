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

import numpy as np

from akg.utils import kernel_exec as utils
from akg.ops.math.ascend import acosh
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian


def acosh_run(x_shape, x_dtype, attrs):
    """run function for dsl function acosh."""
    shapes = [x_shape]
    dtypes = [x_dtype]
    mod = utils.op_build_test(acosh, shapes, dtypes,
                              kernel_name="acosh", attrs=attrs)
    bench_mark, input_datas, output = gen_data(x_dtype, x_shape)
    output = utils.mod_launch(mod, input_datas + [output], expect=bench_mark)
    rtol, atol = get_rtol_atol("acosh", x_dtype)
    compare_res = compare_tensor(output, bench_mark, rtol=rtol, atol=atol)
    return input_datas, output, bench_mark, compare_res


def gen_data(dtype, shape):
    """Generate data for testing the op"""
    input_data = np.abs(random_gaussian(shape).astype(dtype)) + 1
    expect = archcosh_compute(input_data)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, [input_data], output


def archcosh_compute(input_data):
    """implement asinh in numpy"""
    dtype = input_data.dtype
    if dtype == "float16":
        # To keep same as dsl
        input_data = input_data.astype("float32")

    res = np.arccosh(input_data)

    if res.dtype != dtype:
        res = res.astype(dtype)
    return res
