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

"""operator dsl fuction: apply_ada_max"""

import akg
import akg.utils as utils
from akg import topi, tvm
from akg.ops.math import Reciprocal
from akg.utils.dsl_create import TensorUtils, neg_one_const, one_const
from akg.utils.format_transform import get_shape


def _apply_ada_max_compute(var, m, v, grad, lr, beta1, beta1_power, beta2, epsilon):
    """Compute ada_max."""
    # cast to float32 for improved accuracy
    inp_dtype = var.dtype
    if inp_dtype == 'float16':
        var = topi.cast(var, 'float32')
        m = topi.cast(m, 'float32')
        v = topi.cast(v, 'float32')
        lr = topi.cast(lr, 'float32')
        beta1_power = topi.cast(beta1_power, 'float32')
        beta1 = topi.cast(beta1, 'float32')
        beta2 = topi.cast(beta2, 'float32')
        grad = topi.cast(grad, 'float32')
    epsilon = tvm.const(epsilon, 'float32')

    # m += (grad - m) * (1 - beta1)
    rhs = tvm.compute(beta1.shape, lambda *i: beta1(*i) * neg_one_const("float32"))
    rhs = tvm.compute(rhs.shape, lambda *i: rhs(*i) + one_const("float32"))
    lhs = topi.subtract(grad, m)
    rhs = tvm.compute(lhs.shape, lambda *i: lhs(*i) * rhs[0])
    m = topi.add(m, rhs)

    # v = max(beta2*v, abs(grad))
    lhs = tvm.compute(v.shape, lambda *i: v(*i) * beta2[0])
    rhs = topi.abs(grad)
    v = topi.maximum(lhs, rhs)

    # var -= lr / (1 - beta1_power) * (m / (v + epsilon))
    # lr * m / (1 - beta1_power) * (v + epsilon)
    # v + epsilon
    rhs = tvm.compute(v.shape, lambda *i: v(*i) + epsilon)
    # 1 - beta1_power
    lhs = tvm.compute(beta1_power.shape, lambda *i: beta1_power(*i) * neg_one_const("float32"))
    lhs = tvm.compute(lhs.shape, lambda *i: lhs(*i) + one_const("float32"))
    # (1 - beta1_power) * (v + epsilon)
    rhs = tvm.compute(rhs.shape, lambda *i: rhs(*i) * lhs[0])
    # lr * m
    lhs = tvm.compute(m.shape, lambda *i: m(*i) * lr[0])
    # lr * m / (1 - beta1_power) * (v + epsilon)
    rhs = Reciprocal(rhs)
    rhs = topi.multiply(lhs, rhs)
    var = topi.subtract(var, rhs)

    if inp_dtype == 'float16':
        var = topi.cast(var, inp_dtype)
        m = topi.cast(m, inp_dtype)
        v = topi.cast(v, inp_dtype)

    return var, m, v


def _check_inputs(var, m, v, grad, lr, beta1, beta1_power, beta2, epsilon):
    """Check op inputs"""
    # check dtype
    utils.ops_dtype_check(var.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    for i in (m, v, grad, beta1_power, lr, beta1, beta2):
        utils.elemwise_dtype_check(var.dtype, i.dtype)

    # check shape
    for i in (m, v, grad):
        utils.elemwise_shape_check(var.shape, i.shape)
    for i in (beta1_power, lr, beta1, beta2):
        if tuple(get_shape(i)) != (1,):
            raise RuntimeError("beta1_power, lr, beta1 and beta2 only support scalar tensor.")

    # check value
    if epsilon <= 0:
        raise ValueError("epsilon should be greater than zero.")


@utils.check_input_type(*([akg.tvm.tensor.Tensor] * 8), float, (str, type(None)))
def apply_ada_max(var, m, v, grad, lr, beta1, beta1_power, beta2, epsilon, target=utils.CCE):
    """
    Update var according to the AdaMax algorithm.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- max(beta2 * v_{t-1}, abs(g))
    variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float32.
        m (tvm.tensor.Tensor): A tensor of same shape and type as var.
        v (tvm.tensor.Tensor): A tensor of same shape and type as var.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.
        lr (tvm.tensor.Tensor): Learning rate, a scalar tensor of same type as var.
        beta1 (tvm.tensor.Tensor): A scalar tensor of same type as var, 0.0 <= beta1 <= 1.0.
        beta1_power (tvm.tensor.Tensor): The value of :math:`beta1^t`, a scalar tensor of same type as var.
        beta2 (tvm.tensor.Tensor): A scalar tensor of same type as var, 0.0 <= beta2 <= 1.0.
        epsilon (float): A small value to prevent division by 0.

    Returns:
        tvm.tensor.Tensor, Updated var.
        tvm.tensor.Tensor, Updated m.
        tvm.tensor.Tensor, Updated v.
    """

    _check_inputs(var, m, v, grad, lr, beta1, beta1_power, beta2, epsilon)

    out_var, out_m, out_v = _apply_ada_max_compute(var, m, v, grad, lr, beta1, beta1_power, beta2, epsilon)

    # reuse var, m and v
    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    out_m, binds_info2 = TensorUtils.inplace_set(m, out_m, "m_buf")
    out_v, binds_info3 = TensorUtils.inplace_set(v, out_v, "v_buf")
    binds_info.update(binds_info2)
    binds_info.update(binds_info3)
    attrs = {utils.BINDS: binds_info}
    return out_var, out_m, out_v, attrs
