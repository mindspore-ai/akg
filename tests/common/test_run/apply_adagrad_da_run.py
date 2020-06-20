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
from test_op import apply_adagrad_da


def apply_adagrad_da_execute(shape, dtype, attrs=None):
    mod = apply_adagrad_da_compile(shape, dtype, attrs)
    exp_output, inputs, args = gen_data(shape, dtype)
    acu_output = utils.mod_launch(mod, args, outputs=(0, 1, 2), expect=exp_output)
    rtol, atol = get_rtol_atol("apply_adagrad_da", dtype)
    results = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), acu_output, exp_output))
    return inputs, acu_output, exp_output, all(results)


def gen_data(shape, dtype):
    var = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    mg = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    ms = np.abs(random_gaussian(shape, miu=1, sigma=0.1).astype(dtype))
    grad = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    lr = np.random.rand(1).astype(dtype)
    l1 = random_gaussian((1,), miu=1, sigma=0.1).astype(dtype)
    l2 = random_gaussian((1,), miu=1, sigma=0.1).astype(dtype)
    global_step = np.random.randint(10, size=(1,), dtype="int32")

    inputs = [var, mg, ms, grad, lr, l1, l2, global_step]

    mg_out = mg + grad
    ms_out = ms + grad * grad
    if l1 > 0:
        tmp_val = np.sign(mg_out) * np.maximum(np.abs(mg_out) - l1 * global_step,
                                               np.zeros_like(mg_out))
    else:
        tmp_val = mg_out
    var_out = (-lr * tmp_val) / (l2 * global_step * lr + np.sqrt(ms_out))

    exp_output = (var_out, mg_out, ms_out)
    args = inputs

    return exp_output, inputs, args


def apply_adagrad_da_compile(shape, dtype, attrs, tuning=False):
    shapes = [shape, shape, shape, shape, (1,), (1,), (1,), (1,)]
    dtypes = [dtype, dtype, dtype, dtype, dtype, dtype, dtype, "int32"]
    return utils.op_build_test(apply_adagrad_da.apply_adagrad_da, shapes, dtypes,
                               kernel_name="apply_adagrad_da", attrs=attrs, tuning=tuning)
