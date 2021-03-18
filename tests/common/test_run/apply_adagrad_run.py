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
from tests.common.test_op import apply_adagrad
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def apply_adagrad_execute(shape, dtype, update_slots, attrs=None):
    if attrs is None:
        attrs = {}
    mod = apply_adagrad_compile(shape, dtype, update_slots, attrs)
    exp_output, inputs, args = gen_data(dtype, update_slots, shape)
    acu_output = utils.mod_launch(mod, args, outputs=(-2, -1), expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("apply_adagrad", dtype)
    results = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), acu_output, exp_output))
    return inputs, acu_output, exp_output, all(results)


def gen_data(dtype, update_slots, shape):
    var = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    # accum must be greater than 0
    epsilon = 1e-7 if dtype == np.float32 else 1e-3
    accum = np.abs(random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)) + epsilon
    lr = random_gaussian((1,), miu=1, sigma=0.1).astype(dtype)
    grad = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    inputs = [var, accum, lr, grad]
    exp_output = apply_adagrad_compute(var, accum, lr, grad, update_slots)
    outputs = [np.full(e.shape, np.nan, dtype) for e in exp_output]
    args = [*inputs, *outputs]

    return exp_output, inputs, args


def apply_adagrad_compile(shape, dtype, update_slots, attrs, kernel_name="apply_adagrad", tuning=False):
    shapes = [shape, shape, (1,), shape]
    dtypes = [dtype] * len(shapes)
    return utils.op_build_test(apply_adagrad.apply_adagrad, shapes, dtypes, [update_slots],
                               kernel_name=kernel_name, attrs=attrs, tuning=tuning)


def apply_adagrad_compute(var, accum, lr, grad, update_slots):
    dtype = var.dtype
    compute_dtype = "float32"
    if dtype != compute_dtype:
        var, accum, lr, grad = [t.astype(compute_dtype) for t in [var, accum, lr, grad]]
    accum_out = accum + grad * grad if update_slots else accum
    var_out = var - (lr * grad / np.sqrt(accum_out))
    exp_output = [var_out, accum_out]
    if compute_dtype != dtype:
        exp_output = [t.astype(dtype) for t in exp_output]
    return exp_output
