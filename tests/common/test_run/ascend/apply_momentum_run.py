# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from akg.ops.optimizers.ascend import ApplyMomentum
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def apply_momentum_run(shape, dtype, use_nesterov=False, grad_scale=1.0, lr_mat=0.1, momt_mat=0.9, attrs=None):
    """
    run function for dsl function apply_momentum.
    """
    lr = np.full((1,), lr_mat).astype(dtype)
    momt = np.full((1,), momt_mat).astype(dtype)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(ApplyMomentum, [shape, shape, shape, lr.shape, momt.shape],
                                  [dtype, dtype, dtype, dtype, dtype], [use_nesterov, grad_scale], kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            accum_exp, accum, expect, grad, var = gen_data(dtype, lr, momt, shape, use_nesterov, grad_scale)
            fake_output = np.full(shape, np.nan, dtype)
            return mod, (expect, accum_exp), {"args": (var, grad, accum, lr, momt, fake_output), 'outputs': (0, 2, -1), 'tuning': False}
        else:
            return mod
    else:
        mod = utils.op_build_test(ApplyMomentum, [shape, shape, shape, lr.shape, momt.shape],
                                  [dtype, dtype, dtype, dtype, dtype], [use_nesterov, grad_scale], kernel_name='apply_momentum',
                                  attrs=attrs)
        accum_exp, accum, expect, grad, var = gen_data(dtype, lr, momt, shape, use_nesterov, grad_scale)
        fake_output = np.full(shape, np.nan, dtype)
        var_update, accum_update, _ = utils.mod_launch(mod, (var, grad, accum, lr, momt, fake_output),
                                                       outputs=(0, 2, -1), expect=expect)
        rtol, atol = get_rtol_atol("apply_momentum", dtype)
        expects = (expect, accum_exp)
        outputs = (var_update, accum_update)
        return var, outputs, expects, compare_tensor(outputs, expects, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, lr, momt, shape, use_nesterov, grad_scale):
    """
    Generate data for testing the op
    """
    var = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    grad = random_gaussian(shape, miu=3, sigma=0.3).astype(dtype)
    accum = random_gaussian(shape, miu=4, sigma=0.3).astype(dtype)
    grad_compute_value = grad * grad_scale
    momt_update = momt[0] * accum + grad_compute_value
    if use_nesterov == False:
        var_update = var - lr[0] * momt_update
    else:
        var_update = var - lr[0] * grad_compute_value - momt_update * lr[0] * momt[0]
    expect = var_update
    return momt_update, accum, expect, grad, var
