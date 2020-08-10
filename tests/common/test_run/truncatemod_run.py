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

from akg.utils import kernel_exec as utils
from tensorio import compare_tensor
import numpy as np
from gen_random import random_gaussian

from test_op import truncatemod
from base import get_rtol_atol


def truncatemod_run(shape1, shape2, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(truncatemod.truncatemod, [shape1, shape2], [dtype, dtype], kernel_name=kernel_name,
                                  attrs=attrs, dump_code=True, tuning=t)
        if t:
            expect, input1, input2, output = gen_data(dtype, shape1, shape2)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        expect, input1, input2, output = gen_data(dtype, shape1, shape2)
        mod = utils.op_build_test(truncatemod.truncatemod, [shape1, shape2], [dtype, dtype], kernel_name="truncatemod",
                                  attrs=attrs, dump_code=True)
        output = utils.mod_launch(mod, (input1, input2, output), expect=expect)
        rtol, atol = get_rtol_atol("truncatemod", dtype)
        res = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)
        return (input1, input2), output, expect, res


def truncatemod_compute(x, y):
    dtype = x.dtype
    if dtype != "float32":
        x = x.astype("float32")
        y = y.astype("float32")
    expect = (x - y*np.trunc(x/y))

    if expect.dtype != dtype:
        expect = expect.astype(dtype)

    return expect


def gen_data(dtype, shape1, shape2):
    input1 = random_gaussian(shape1).astype(dtype)
    input2 = random_gaussian(shape2).astype(dtype)
    # mod 0 is undefined
    input2 = np.select(input2 == 0, np.ones_like(input2), input2)
    if utils.product_is_mini():
        # If the value of input2 is too small, input1/input2 will be some overflow
        lower_bound = 1e-3
        input2 = np.select([input2 >= 0, input2 < 0], [np.maximum(input2, lower_bound),
                                                       np.minimum(input2, -lower_bound)])
    expect = truncatemod_compute(input1, input2)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input1, input2, output
