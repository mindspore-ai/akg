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
from tensorio import compare_tensor
import numpy as np
from test_op.apply_proximal_gradient_descent import apply_proximal_gradient_descent
from base import get_rtol_atol
from gen_random import random_gaussian


def apply_proximal_gradient_descent_run(shape, dtype, attrs=None):
    """run function for dsl function apply_proximal_gradient_descent."""
    scalar_shape = (1,)
    var_shape, delta_shape = shape, shape
    alpha_shape, l1_shape, l2_shape = [scalar_shape] * 3
    shapes = [var_shape, alpha_shape, l1_shape, l2_shape, delta_shape]
    dtypes = [dtype] * 5
    mod = utils.op_build_test(apply_proximal_gradient_descent, shapes, dtypes, 
                              kernel_name='apply_proximal_gradient_descent', attrs=attrs)
    expect, (var, alpha, l1, l2, delta) = gen_data(dtype, shape)
    output = utils.mod_launch(mod, (var, alpha, l1, l2, delta), outputs=(0, ))
    rtol, atol = get_rtol_atol("apply_proximal_gradient_descent", dtype)
    compare_result = compare_tensor(output, expect, rtol=rtol, atol=atol)
    inputs = (var, alpha, l1, l2, delta)
    return inputs, output, expect, compare_result


def apply_proximal_gradient_descent_impl(var, alpha, l1, l2, delta):
    """implement apply_proximal_gradient_descent in numpy"""
    dtype = var.dtype
    if dtype == "float16":
        # to keep same with dsl
        var, alpha, l1, l2, delta = [t.astype("float32") for t in [var, alpha, l1, l2, delta]]
    prox_var = var - alpha * delta
    var_new_l1_gt_0 = np.sign(prox_var) / (1+alpha*l2) * np.maximum(np.abs(prox_var)-alpha*l1, 0)
    var_new_l1_le_0 = prox_var/(1+alpha*l2)
    if l1[0] > 0:
        var_new = var_new_l1_gt_0
    else:
        var_new = var_new_l1_le_0
    if var_new.dtype != dtype:
        var_new = var_new.astype(dtype)
    return var_new


def gen_data(dtype, shape):
    """Generate data for testing the op"""

    # tensors
    var = random_gaussian(shape).astype(dtype)
    delta = random_gaussian(shape).astype(dtype)
    # scalars
    scalar_shape = (1,)
    alpha = np.random.random_sample(scalar_shape).astype(dtype)
    l1 = np.random.randn(*scalar_shape).astype(dtype)
    l2 = np.random.random_sample(scalar_shape).astype(dtype)
    input_data = (var, alpha, l1, l2, delta)

    # expects
    expect = apply_proximal_gradient_descent_impl(var, alpha, l1, l2, delta)
    return expect, input_data
