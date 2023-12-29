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
from akg.ops.math.ascend import approximate_equal
from tests.common.gen_random import random_gaussian


def approximate_equal_run(x_shape, x_dtype, y_shape, y_dtype, tolerance=None, attrs=None):
    shapes = [x_shape, y_shape]
    dtypes = [x_dtype, y_dtype]
    op_attrs = None
    if tolerance:
        op_attrs = [tolerance]
    mod = utils.op_build_test(approximate_equal, shapes, dtypes, op_attrs,
          kernel_name="approximate_equal", attrs=attrs)
    benchMark, inputs, output = gen_data(x_dtype, shapes, tolerance)
    output = utils.mod_launch(mod, inputs + [output], expect=benchMark)
    return inputs, output, benchMark, np.array_equal(output, benchMark)


def gen_data(dtype, shapes, tolerance):
    if len(shapes) != 2:
        raise RuntimeError("inputs num should be 2")
    if not tolerance:
        tolerance = 1e-5
    x = random_gaussian(shapes[0]).astype(dtype)
    # If y is randomly generated, the difference from x will both be 
    # larger than tolerance when tolerance is small.
    y = x + np.random.uniform(low=-2.0*tolerance, high=2.0*tolerance)
    benchMark = np.isclose(x, y, rtol=0.0, atol=tolerance)
    output = np.full(benchMark.shape, 0, bool)
    return benchMark, [x, y], output
