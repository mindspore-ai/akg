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

from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
import numpy as np
from tests.common.test_op.ascend.apply_proximal_adagrad import apply_proximal_adagrad
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.test_run.ascend.apply_proximal_gradient_descent_run import apply_proximal_gradient_descent_impl
from akg.utils.kernel_exec import product_is_mini

def apply_proximal_adagrad_run(shape, dtype, attrs=None):
    """run function for dsl function apply_proximal_adagrad."""
    scalar_shape = (1,)
    var_shape, accum_shape, grad_shape = [shape] * 3
    lr_shape, l1_shape, l2_shape = [scalar_shape] * 3
    shapes = [var_shape, accum_shape, lr_shape, l1_shape, l2_shape, grad_shape]
    dtypes = [dtype] * 6
    mod = utils.op_build_test(apply_proximal_adagrad, shapes, dtypes,
                              kernel_name='apply_proximal_adagrad', attrs=attrs)
    expects, (var, accum, lr, l1, l2, grad) = gen_data(dtype, shape)
    outputs = utils.mod_launch(mod, (var, accum, lr, l1, l2, grad), outputs=(0, 1))
    rtol, atol = get_rtol_atol("apply_proximal_adagrad", dtype)
    compare_result = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), outputs, expects))
    inputs = (var, accum, lr, l1, l2, grad)
    return inputs, outputs, expects, all(compare_result)


def _apply_proximal_adagrad_compute(var, accum, lr, l1, l2, grad):
    """implement apply_proximal_adagrad in numpy"""
    dtype = var.dtype
    if dtype == "float16":
        # to keep same with dsl
        var, accum, lr, l1, l2, grad = [t.astype("float32") for t in [var, accum, lr, l1, l2, grad]]
    accum_new = accum + grad * grad
    if product_is_mini():
        accum_new_rsqrt = np.reciprocal(np.sqrt(accum_new))
        ada_lr = lr * accum_new_rsqrt
    else:
        ada_lr = np.divide(lr, np.sqrt(accum_new))
    var_new = apply_proximal_gradient_descent_impl(var, ada_lr, l1, l2, grad)

    if var_new.dtype != dtype:
        var_new = var_new.astype(dtype)
    return var_new, accum_new


def gen_data(dtype, shape):
    """Generate data for testing the op"""

    # tensors
    var = random_gaussian(shape).astype(dtype)
    accum = np.abs(random_gaussian(shape).astype(dtype))
    grad = random_gaussian(shape).astype(dtype)
    # scalars
    scalar_shape = (1,)
    lr = np.random.random_sample(scalar_shape).astype(dtype)
    l1 = np.random.randn(*scalar_shape).astype(dtype)
    l2 = np.random.random_sample(scalar_shape).astype(dtype)

    input_data = (var, accum, lr, l1, l2, grad)
    expect = _apply_proximal_adagrad_compute(var, accum, lr, l1, l2, grad)
    return expect, input_data

