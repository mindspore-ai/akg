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
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from tests.common.test_op import apply_add_sign


def apply_add_sign_execute(shape, dtype, attrs=None):
    mod = apply_add_sign_compile(shape, dtype, attrs)
    exp_output, inputs, args = gen_data(shape, dtype)
    acu_output = utils.mod_launch(mod, args, outputs=(0, 1), expect=exp_output)
    rtol, atol = get_rtol_atol("apply_add_sign", dtype)
    results = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), acu_output, exp_output))
    return inputs, acu_output, exp_output, all(results)


def gen_data(shape, dtype):
    var = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    m = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    grad = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    lr = np.random.rand(1).astype(dtype)
    alpha = np.random.rand(1).astype(dtype)
    sign_decay = np.random.rand(1).astype(dtype)
    beta = np.random.rand(1).astype(dtype)

    inputs = [var, m, grad, lr, alpha, sign_decay, beta]

    m_out = m * beta + grad * (1 - beta)
    var_out = var - lr * (alpha + sign_decay * np.sign(grad) * np.sign(m)) * grad

    exp_output = (var_out, m_out)
    args = inputs

    return exp_output, inputs, args


def apply_add_sign_compile(shape, dtype, attrs, kernel_name="apply_add_sign", tuning=False):
    shapes = [shape, shape, shape, (1,), (1,), (1,), (1,)]
    dtypes = [dtype] * len(shapes)
    return utils.op_build_test(apply_add_sign.apply_add_sign, shapes, dtypes,
                               kernel_name=kernel_name, attrs=attrs, tuning=tuning)
