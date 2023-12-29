# Copyright 2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: lamb apply optimizer assign"""
import akg.tvm
import akg.utils as utils
from akg import topi
from akg.ops.math.log import log
from akg.ops.math.neg import neg
from akg.ops.math.divide import divide
from akg.ops.math.exp import exp
from akg.utils.format_transform import get_shape
from akg.utils import custom_tiling as ct_util

lamb_apply_optimizer_assign_set_dim_map = {
    "((4096, 1024), 'float32')": ((1024, 1024),),
    "((1024, 1024), 'float32')": ((1024, 1024),),
    "((1024, 4096), 'float32')": ((1024, 1024),),
    "((512, 1024), 'float32')": ((1024, 1024),),
    "((30522, 1024), 'float32')": ((4352, 4352),),
}


def lamb_apply_optimizer_assign_set_dim_func(data):
    """set dim info for attr."""
    shape = get_shape(data)
    hash_key = str((tuple(shape), data.dtype))
    return ct_util.set_dims_by_key(hash_key, lamb_apply_optimizer_assign_set_dim_map), hash_key


def pow_compute(input_x, input_y, data):
    """
    :param input_x:
    :param input_y:
    :return: exp(input_y * ln(input_x))
    """

    input_x_broadcast = akg.lang.ascend.broadcast(input_x, data.shape)
    log_value = log(input_x_broadcast, utils.CCE)
    mul_value = topi.multiply(input_y, log_value)
    res = exp(mul_value, utils.CCE)

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          (str, type(None)))
def LambApplyOptimizerAssign(grad, input_v, input_m, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon, steps, do_use_weight,
                             weight_decay_rate):

    # compute next_v
    square_grad = topi.multiply(grad, grad)

    # mul_3
    mul_3_result = topi.multiply(square_grad, one_minus_beta_2)

    # mul_2
    mul_2_result = topi.multiply(input_v, beta_2)

    # compute: next_v = (multiply(self.beta_2, v) + multiply(1.0 - self.beta_2, square(grad)))
    next_v = topi.add(mul_2_result, mul_3_result)

    # compute next_m
    mul_0_result = topi.multiply(input_m, beta_1)

    # mul_1
    mul_1_result = topi.multiply(grad, one_minus_beta_1)

    # compute: next_m = (multiply(self.beta_1, m) + multiply(1.0 - self.beta_1, grad))
    next_m = topi.add(mul_0_result, mul_1_result)

    const_one = akg.tvm.const(1.0, input_v.dtype)

    # compute: beta1_correction = (1 - self.beta_1 ** steps)
    beta_1_steps = pow_compute(beta_1, steps, grad)
    neg_beta_1_step = neg(beta_1_steps, utils.CCE)
    beta1_correction = topi.add(neg_beta_1_step, const_one)

    # compute: beta2_correction = (1 - self.beta_2 ** steps)
    beta_2_steps = pow_compute(beta_2, steps, grad)
    neg_beta_2_step = neg(beta_2_steps, utils.CCE)
    beta2_correction = topi.add(neg_beta_2_step, const_one)

    # compute: next_m_unbiased = next_m / beta1_correction
    next_m_unbiased = divide(next_m, beta1_correction, utils.CCE)
    # compute: next_v_unbiased = next_v / beta2_correction
    next_v_unbiased = divide(next_v, beta2_correction, utils.CCE)

    # compute update
    sqrt_next_v = topi.sqrt(next_v_unbiased)
    # add_2
    add_2_result = topi.add(sqrt_next_v, epsilon)
    # compute: update = next_m / (sqrt(next_v) + self.epsilon)
    update = divide(next_m_unbiased, add_2_result, utils.CCE)

    # compute do_use_weight_decay
    do_use_weight_mul = topi.multiply(input_param, weight_decay_rate)
    do_use_weight_decay = topi.multiply(do_use_weight_mul, do_use_weight)
    update = topi.add(do_use_weight_decay, update)

    attrs = {'enable_auto_inline': False}

    dim_info, _ = lamb_apply_optimizer_assign_set_dim_func(grad)
    if dim_info != "":
        attrs["dim"] = dim_info

    return update, next_v, next_m, attrs
