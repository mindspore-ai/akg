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
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from tests.common.test_op.ascend.apply_gradient_descent import apply_gradient_descent


def apply_gradient_descent_run(shape, dtype, attrs=None):
    """run function for dsl function apply_gradient_descent."""
    shapes = [shape, (1,), shape]
    dtypes = [dtype] * len(shapes)

    mod = utils.op_build_test(apply_gradient_descent, shapes, dtypes,
                              kernel_name='apply_gradient_descent', attrs=attrs)
    inputs, expect, args = gen_data(shape, dtype)
    output = utils.mod_launch(mod, args, outputs=(0,), expect=expect)
    rtol, atol = get_rtol_atol("apply_gradient_descent", dtype)
    result = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)
    return inputs, output, expect, result


def gen_data(shape, dtype):
    """Generate data for testing the op."""
    var = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    alpha = random_gaussian((1,), miu=3, sigma=0.3).astype(dtype)
    delta = random_gaussian(shape, miu=4, sigma=0.3).astype(dtype)
    inputs = [var, alpha, delta]
    expect = var - alpha * delta
    args = inputs
    return inputs, expect, args
