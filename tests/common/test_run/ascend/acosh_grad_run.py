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
from akg.ops.math.ascend import AcoshGrad
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian


def acosh_grad_run(shape, dtype, attrs):
    """run function for dsl function acosh_grad."""
    shapes = [shape, shape]
    dtypes = [dtype, dtype]
    mod = utils.op_build_test(AcoshGrad, shapes, dtypes,
                              kernel_name="acosh_grad", attrs=attrs)
    bench_mark, inputs, output = gen_data(dtype, shape)
    output = utils.mod_launch(mod, inputs + [output], expect=bench_mark)
    rtol, atol = get_rtol_atol("acosh_grad", dtype)
    compare_res = compare_tensor(output, bench_mark, rtol=rtol, atol=atol)
    return inputs, output, bench_mark, compare_res


def gen_data(dtype, shape):
    """Generate data for testing the op"""
    y = random_gaussian(size=shape).astype(dtype)
    if dtype == "float16":
        # If the value of y is too small, there will be some overflow
        lower_bound = 1e-3
        y = np.select([y >= 0, y < 0], [np.maximum(y, lower_bound), np.minimum(y, -lower_bound)])
    dy = random_gaussian(size=shape).astype(dtype)

    expect = _acosh_grad_compute(y, dy)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, [y, dy], output


def _acosh_grad_compute(y, dy):
    """implement acosh grad in numpy"""
    dtype = y.dtype
    if dtype == "float16":
        y = y.astype("float32")
        dy = dy.astype("float32")

    res = np.divide(dy, np.sinh(y))

    if res.dtype != dtype:
        res = res.astype(dtype)
    return res
