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

from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
import numpy as np
from tests.common.test_op.apply_ftrl import apply_ftrl
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def apply_ftrl_run(shape, dtype, attrs=None):
    """run function for dsl function apply_ftrl."""
    scalar_shape = (1,)
    var_shape, accum_shape, linear_shape, grad_shape = [shape] * 4
    lr_shape, l1_shape, l2_shape, lr_power_shape = [scalar_shape] * 4
    shapes = [var_shape, accum_shape, linear_shape, grad_shape,
              lr_shape, l1_shape, l2_shape, lr_power_shape]
    dtypes = [dtype] * 9
    mod = utils.op_build_test(apply_ftrl, shapes, dtypes, kernel_name='apply_ftrl', attrs=attrs)
    expects, (var, accum, linear, grad), (lr, l1, l2, lr_power) = gen_data(dtype, shape)
    outputs = utils.mod_launch(mod, (var, accum, linear, grad, lr, l1, l2, lr_power),
                               outputs=(0, 1, 2))
    rtol, atol = get_rtol_atol("apply_ftrl", dtype)
    compare_result = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), outputs, expects))
    inputs = (var, accum, linear, grad, lr, l1, l2, lr_power)
    return inputs, outputs, expects, all(compare_result)


def gen_data(dtype, shape, with_l2_shrinkage=False):
    """Generate data for testing the op"""

    # tensors
    var = random_gaussian(shape).astype(dtype)
    accum = np.abs(random_gaussian(shape).astype(dtype))
    linear = random_gaussian(shape).astype(dtype)
    grad = random_gaussian(shape).astype(dtype)
    tensors = [var, accum, linear, grad]

    # scalars
    scalar_shape = (1,)
    lr = np.random.random_sample(scalar_shape).astype(dtype)
    l1 = np.random.random_sample(scalar_shape).astype(dtype)
    l2 = np.random.random_sample(scalar_shape).astype(dtype)
    lr_power = np.array([0.5], dtype)
    if with_l2_shrinkage:
        l2_shrinkage = np.random.random_sample(scalar_shape).astype(dtype)
        scalars = [lr, l1, l2, l2_shrinkage, lr_power]
    else:
        scalars = [lr, l1, l2, lr_power]

    # expects
    expects = apply_ftrl_impl(tensors, scalars, with_l2_shrinkage)

    return expects, tensors, scalars


def apply_ftrl_impl(tensors, scalars, with_l2_shrinkage=False):
    """implement Ftrl-proximal Optimization algorithm in numpy"""
    var, accum, linear, grad = tensors
    if with_l2_shrinkage:
        lr, l1, l2, l2_shrinkage, lr_power = scalars
    else:
        lr, l1, l2, lr_power = scalars
    dtype = var.dtype

    compute_dtype = dtype
    if dtype == "float16":
        # to keep same with dsl
        compute_dtype = "float32"
        var, accum, linear, grad = [t.astype(compute_dtype) for t in tensors]
        if with_l2_shrinkage:
            lr, l1, l2, l2_shrinkage, lr_power = [s.astype(compute_dtype) for s in scalars]
        else:
            lr, l1, l2, lr_power = [s.astype(compute_dtype) for s in scalars]

    accum_new = accum + grad * grad
    if with_l2_shrinkage:
        grad_shrinkage = grad + 2 * l2_shrinkage * var
    else:
        grad_shrinkage = grad
    linear_new = linear + grad_shrinkage - (np.power(accum_new, -lr_power) - np.power(accum, -lr_power)) / lr * var
    linear_new_clip = np.clip(linear_new, -l1[0], l1[0])
    x = linear_new_clip - linear_new
    y = np.power(accum_new, -lr_power) / lr + 2 * l2
    var_new = np.divide(x, y)
    expects = [var_new, accum_new, linear_new]

    if compute_dtype != dtype:
        expects = [t.astype(dtype) for t in expects]
    return expects
