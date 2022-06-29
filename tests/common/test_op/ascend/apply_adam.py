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

"""operator dsl function: apply_adam"""
import akg.tvm
import akg.topi
from akg.utils.format_transform import get_shape
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils
from akg.ops.math import divide, sqrt


def _apply_adam_compute(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_nesterov=False):
    """Compute for adam algorithm"""

    shape = var.shape

    # m_new <- m + (1-beta1)*(grad - m)
    m_new = akg.tvm.compute(shape, lambda *indice: m(*indice) + (1-beta1[0])*(grad(*indice)-m(*indice)), name="m_new")

    # v_new <- v + (1-beta2)*(grad*grad-v)
    v_new = akg.tvm.compute(shape, lambda *indice: v(*indice) + (1-beta2[0])*(grad(*indice)*grad(*indice)-v(*indice)),
                            name="v_new")

    # lr_t <- lr*sqrt(1-beta2_power)/(1-beta1_power)
    one_const = akg.tvm.const(1, var.dtype)
    sqrt_value_beta2 = sqrt(akg.topi.subtract(one_const, beta2_power), target=utils.CCE)
    lr_mul_sqrt_value = akg.topi.multiply(lr, sqrt_value_beta2)
    sub_value_beta1 = akg.topi.subtract(one_const, beta1_power)
    lr_t = divide(lr_mul_sqrt_value, sub_value_beta1, target=utils.CCE)

    # if use_nersterov: var_new <- var - lr_t*(m_new*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_new))
    # if not use_nersterov: var_new <- var - lr_t*m_new / (epsilon + sqrt(v_new))
    if use_nesterov:
        lr_t_mul_m_new = akg.tvm.compute(shape, lambda *indice:
                                         lr_t[0]*(m_new(*indice)*beta1[0] + (1-beta1[0])*grad(*indice)),
                                         name="lr_t_mul_m_new")
    else:
        lr_t_mul_m_new = akg.tvm.compute(shape, lambda *indice: lr_t[0] * m_new(*indice), name="lr_t_mul_m_new")
    sqrt_value_v_new = sqrt(v_new, target=utils.CCE)
    epsilon_add_sqrt_value = akg.topi.add(epsilon, sqrt_value_v_new)
    div_value = divide(lr_t_mul_m_new, epsilon_add_sqrt_value, target=utils.CCE)
    var_new = akg.topi.subtract(var, div_value)

    return var_new, m_new, v_new


@utils.check_input_type(*([akg.tvm.tensor.Tensor]*10 + [(bool, type(None))]), (str, type(None)))
def apply_adam(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_nesterov=False, target=utils.CCE):
    """
    Adam and Nadam optimization algorithm.

    Note:
        lr_t = lr*sqrt(1-beta2_power)/(1-beta1_power)
        m_new = m + (1-beta1)*(grad-m)
        v_new = v + (1-beta2)*(grad*grad-v)
        if user_nesterov == True:
            var_new = var - lr_t*(m_new*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_new))
        else:
            var_new = var - lr_t*m_new / (epsilon + sqrt(v_new))

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float16 or float32.
        m (tvm.tensor.Tensor): The first moment estimate. A tensor of same shape and type as var.
        v (tvm.tensor.Tensor): The second moment estimate. A tensor of same shape and type as var.
        beta1_power (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        beta2_power (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        lr (tvm.tensor.Tensor): The learning rate. A scalar tensor of the same type as `var`.
        beta1(tvm.tensor.Tensor): A tensor with shape (1,) and type is same as var.
        beta2(tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        epsilon(tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.
        use_nesterov(bool): Default value is False. If use_nesterov is True, the Nadam algorithm be implemented,
                            otherwise the adam algorithm be implemented.

    Returns:
        tvm.tensor.Tensor, updated var.
        tvm.tensor.Tensor, updated m.
        tvm.tensor.Tensor, updated v.
    """

    # check shape
    utils.check_shape(var)
    shape = get_shape(var)
    for tensor in (m, v, grad):
        utils.elemwise_shape_check(shape, tensor.shape)
    sclar_shape = (1,)
    for sclar in (beta1_power, beta2_power, lr, beta1, beta2, epsilon):
        utils.elemwise_shape_check(sclar.shape, sclar_shape)

    # check dtype
    dtype = var.dtype
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.FLOAT16, utils.DtypeForDavinci.FLOAT32])
    for tensor in (var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
        utils.elemwise_dtype_check(tensor.dtype, dtype)

    var_new, m_new, v_new = _apply_adam_compute(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad,
                                                use_nesterov)

    # update by inplace
    (var_new, m_new, v_new), binds_info = TensorUtils.inplace_set_tensors([var, m, v], [var_new, m_new, v_new])
    attrs = {utils.BINDS: binds_info}

    return var_new, m_new, v_new, attrs
