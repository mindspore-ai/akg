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
from tests.common.test_op import apply_centered_rms_prop
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def apply_centered_rms_prop_execute(shape, dtype, lr, momentum, rho, epsilon, attrs=None):
    if attrs is None:
        attrs = {}
    mod = apply_centered_rms_prop_compile(shape, dtype, epsilon, attrs)
    exp_output, inputs, args = gen_data(shape, dtype, lr, momentum, rho, epsilon)
    acu_output = utils.mod_launch(mod, args, outputs=(0, 1, 2, 3), expect=exp_output)
    rtol, atol = get_rtol_atol("apply_centered_rms_prop", dtype)
    results = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), acu_output, exp_output))
    return inputs, acu_output, exp_output, all(results)


def gen_data(shape, dtype, lr, momentum, rho, epsilon):
    var = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    grad = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    rho = np.array([rho]).astype(dtype)
    mg = grad * rho
    ms = grad * grad
    mom = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    lr = np.array([lr]).astype(dtype)
    momentum = np.array([momentum]).astype(dtype)
    inputs = [var, mg, ms, mom, grad, lr, momentum, rho]

    if dtype == "float16":
        var, mg, ms, mom, grad, lr, momentum, rho = [x.astype("float32") for x in inputs]

    one = np.array([1.0], dtype=rho.dtype)
    out_mg = rho * mg + (one - rho) * grad
    out_ms = rho * ms + (one - rho) * grad * grad
    out_mom = momentum * mom + lr * grad / np.sqrt(out_ms - out_mg * out_mg + epsilon)
    out_var = var - out_mom

    exp_output = (out_var, out_mg, out_ms, out_mom)
    if dtype != out_var.dtype:
        exp_output = tuple([x.astype(dtype) for x in exp_output])
    args = inputs

    return exp_output, inputs, args


def apply_centered_rms_prop_compile(shape, dtype, epsilon, attrs, tuning=False):
    shapes = [shape, shape, shape, shape, shape, (1,), (1,), (1,)]
    dtypes = [dtype] * len(shapes)
    op_attrs = [epsilon]
    return utils.op_build_test(apply_centered_rms_prop.apply_centered_rms_prop, shapes, dtypes, op_attrs=op_attrs,
                               kernel_name="apply_centered_rms_prop", attrs=attrs, tuning=tuning)
