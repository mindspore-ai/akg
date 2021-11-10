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

"""apply_rms_prop_mixed_precision_run"""

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import apply_rms_prop
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def apply_rms_prop_mixed_precision_run(shape, dtype, lr, momentum, rho, epsilon, attrs=None):
    """run function for dsl function apply_rms_prop_mixed_precision."""
    if attrs is None:
        attrs = {}

    dtype = dtype.lower()
    shapes = [shape, shape, shape, shape, (1,), (1,), (1,)]
    types = [dtype, dtype, dtype, dtype, dtype, dtype, dtype]
    op_attrs = [epsilon]

    mod = utils.op_build_test(apply_rms_prop.apply_rms_prop_mixed_precision, shapes, types,
                              op_attrs=op_attrs, kernel_name="apply_rms_prop_mixed_precision", attrs=attrs)
    inputs, expects, args = gen_data(shape, dtype, lr, momentum, rho, epsilon)
    outputs = utils.mod_launch(mod, args, outputs=(0, -1, 1, 2), expect=expects)

    # output type: fp32, fp16, fp32, fp32
    precision = [get_rtol_atol("apply_rms_prop", e.dtype) for e in expects]
    results = list(map(lambda x, y, p: compare_tensor(x, y, rtol=p[0], atol=p[1]), outputs, expects, precision))

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

    expects = (var_update, var_update.astype("float16"), ms_update, mom_update)
    outputs = np.full(var_update.shape, np.nan, "float16")
    args = [*inputs, outputs]

    return inputs, expects, args
