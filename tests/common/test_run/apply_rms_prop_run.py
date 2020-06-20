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

"""apply_rms_prop_run"""

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import apply_rms_prop
from base import get_rtol_atol
from gen_random import random_gaussian


def apply_rms_prop_run(shape, dtype, lr, momentum, rho, epsilon, attrs=None):
    """run function for dsl function apply_rms_prop."""
    if attrs is None:
        attrs = {}

    dtype = dtype.lower()
    shapes = [shape, shape, shape, shape, (1,), (1,), (1,)]
    types = [dtype, dtype, dtype, dtype, dtype, dtype, dtype]
    op_attrs = [epsilon]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(apply_rms_prop.apply_rms_prop, shapes, types,
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            _, expects, args = gen_data(shape, dtype, lr, momentum, rho, epsilon)
            return mod, expects, args
        return mod

    mod = utils.op_build_test(apply_rms_prop.apply_rms_prop, shapes, types,
                              op_attrs=op_attrs, kernel_name="apply_rms_prop", attrs=attrs)
    inputs, expects, args = gen_data(shape, dtype, lr, momentum, rho, epsilon)
    outputs = utils.mod_launch(mod, args, outputs=(0, 1, 2), expect=expects)

    rtol, atol = get_rtol_atol("apply_rms_prop", dtype)
    results = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), outputs, expects))
    return inputs, outputs, expects, all(results)


def gen_data(shape, dtype, lr, momentum, rho, epsilon):
    """Generates input, output and expect data."""
    var = random_gaussian(shape, miu=10, sigma=1.0).astype(dtype)
    ms = np.abs(random_gaussian(shape, miu=4, sigma=0.1).astype(dtype))
    mom = random_gaussian(shape, miu=3, sigma=0.3).astype(dtype)
    grad = random_gaussian(shape, miu=3, sigma=0.3).astype(dtype)
    lr = np.array([lr]).astype(dtype)
    momentum = np.array([momentum]).astype(dtype)
    rho = np.array([rho]).astype(dtype)

    inputs = [var, ms, mom, grad, lr, momentum, rho]

    # ms = rho * ms + (1-rho) * grad * grad
    # mom = momentum * mom + lr * grad / sqrt(ms + epsilon)
    # var = var - mom
    one = np.array([1.0]).astype(dtype)
    ms_1 = rho * ms
    ms_2 = (one - rho) * grad * grad
    ms_update = ms_1 + ms_2
    mom_1 = momentum * mom
    mom_2_1 = lr * grad
    mom_2_2 = one / np.sqrt(ms_update + epsilon)
    mom_3 = mom_2_1 * mom_2_2
    mom_update = mom_1 + mom_3
    var_update = var - mom_update

    expects = (var_update, ms_update, mom_update)
    args = inputs
    return inputs, expects, args
