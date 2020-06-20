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
from base import get_rtol_atol
from gen_random import random_gaussian
from tensorio import compare_tensor
from test_op.apply_adadelta import apply_adadelta


def apply_adadelta_run(shape, dtype, epsilon, attrs=None):
    """run function for dsl function apply_adadelta."""
    shapes = [shape, shape, shape, shape, (1,), (1,)]
    dtypes = [dtype] * len(shapes)
    op_attrs = [epsilon]

    mod = utils.op_build_test(apply_adadelta, shapes, dtypes,
                              op_attrs=op_attrs, kernel_name="apply_adadelta", attrs=attrs)
    inputs, expects, args = gen_data(shape, dtype, epsilon)
    outputs = utils.mod_launch(mod, args, outputs=(0, 1, 2), expect=expects)
    rtol, atol = get_rtol_atol("apply_delta", dtype)
    results = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), outputs, expects))
    return inputs, outputs, expects, all(results)


def gen_data(shape, dtype, epsilon):
    """Generate data for testing the op."""
    var = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    accum = np.abs(random_gaussian(shape, miu=1, sigma=0.3).astype(dtype))
    accum_update = np.abs(random_gaussian(shape, miu=1, sigma=0.3).astype(dtype))
    grad = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    lr = np.random.rand(1).astype(dtype)
    rho = np.random.rand(1).astype(dtype)

    inputs = [var, accum, accum_update, grad, lr, rho]

    one = np.array([1]).astype(dtype)
    epsilon = np.array([epsilon]).astype(dtype)

    out_accum = rho * accum + (one - rho) * grad * grad
    update = np.sqrt(accum_update + epsilon) / np.sqrt(out_accum + epsilon) * grad
    out_accum_update = rho * accum_update + (one - rho) * update * update
    out_var = var - update * lr

    expects = [out_var, out_accum, out_accum_update]
    args = inputs

    return inputs, expects, args
