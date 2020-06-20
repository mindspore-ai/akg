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

"""reduce_min_run"""

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import reduce_min
from akg.utils.dsl_create import get_reduce_out_shape
from gen_random import random_gaussian
from base import get_rtol_atol

def reduce_min_run(shape, axis, keepdims, dtype, kernel_name="reduce_min", attrs=None):
    """run function for dsl function reduce_min."""
    if attrs is None:
        attrs = {}

    op_attrs = [axis, keepdims]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(reduce_min.reduce_min, [shape], [dtype], 
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, inputs, output = gen_data(axis, dtype, keepdims, shape)
            return mod, expect, (inputs, output)

        return mod

    mod = utils.op_build_test(reduce_min.reduce_min, [shape], [dtype], 
                              op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs)
    expect, inputs, output = gen_data(axis, dtype, keepdims, shape)
    output = utils.mod_launch(mod, (inputs, output), expect=expect)
    rtol, atol = get_rtol_atol("reduce_min", dtype)
    return inputs, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(axis, dtype, keepdims, shape):
    """Generates input, output and expect data."""
    inputs = random_gaussian(shape, miu=0, sigma=100.0).astype("float16").astype(dtype.lower())
    expect = np.amin(inputs, axis=axis, keepdims=keepdims)
    out_shape = get_reduce_out_shape(shape, axis=axis, keepdims=keepdims)
    output = np.full(out_shape, np.nan, dtype)
    return expect, inputs, output
