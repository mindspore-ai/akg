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
from tests.common.test_op.ascend.apply_adam import apply_adam
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def apply_adam_run(shape, dtype, use_nesterov=None, attrs=None):
    """run function for dsl function apply_adam."""
    scalar_shape = (1,)
    var_shape, m_shape, v_shape = [shape] * 3
    beta1_power_shape, beta2_power_shape, lr_shape, beta1_shape, beta2_shape, epsilon_shape = [scalar_shape] * 6
    grad_shape = shape
    shapes = [var_shape, m_shape, v_shape, beta1_power_shape, beta2_power_shape, lr_shape,
              beta1_shape, beta2_shape, epsilon_shape, grad_shape]
    dtypes = [dtype] * 10
    if use_nesterov is None:
        op_attrs = None
    else:
        op_attrs = [use_nesterov]
    mod = utils.op_build_test(apply_adam, shapes, dtypes, op_attrs, kernel_name='apply_adam', attrs=attrs)
    expects, (var, m, v, grad), (beta1_power, beta2_power, lr, beta1, beta2, epsilon) = gen_data(dtype, shape,
                                                                                                 use_nesterov)
    outputs = utils.mod_launch(mod, (var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad),
                               outputs=(0, 1, 2))
    rtol, atol = get_rtol_atol("apply_adam", dtype)
    compare_result = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), outputs, expects))
    inputs = (var, m, v, grad)
    return inputs, outputs, expects, all(compare_result)


def gen_data(dtype, shape, use_nesterov=False):
    """Generate data for testing the op"""

    # tensors
    var = random_gaussian(shape).astype(dtype)
    m = random_gaussian(shape).astype(dtype)
    v = np.abs(random_gaussian(shape).astype(dtype))
    grad = random_gaussian(shape).astype(dtype)
    tensors = [var, m, v, grad]

    # scalars
    lr = np.array([0.001], dtype)
    beta1 = np.array([0.9], dtype)
    beta2 = np.array([0.999], dtype)
    epsilon = np.array([1e-7], dtype)
    t = np.random.randint(1, 100, size=(1,))
    beta1_power = np.array([beta1 ** t], dtype)
    beta2_power = np.array([beta2 ** t], dtype)
    sclars = [beta1_power, beta2_power, lr, beta1, beta2, epsilon]

    # expects
    lr_coffient = np.sqrt(1.0 - beta2_power)/(1.0 - beta1_power)
    lr_t = lr * lr_coffient
    m_t = m + (1.0 - beta1) * (grad - m)
    v_t = v + (1.0 - beta2) * (grad * grad - v)
    v_t_sqrt = np.sqrt(v_t)
    if use_nesterov:
        var_t = var - (lr_t * (m_t * beta1 + (1.0-beta1)*grad))/(epsilon + v_t_sqrt)
    else:
        var_t = var - (lr_t * m_t)/(epsilon + v_t_sqrt)
    expects = [var_t, m_t, v_t]
    return expects, tensors, sclars
