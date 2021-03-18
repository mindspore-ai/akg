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

"""operator dsl fuction: apply_power_sign"""

import akg
from akg import topi, tvm
from akg.ops.math.exp import exp
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import TensorUtils, neg_one_const, one_const
from akg.utils.format_transform import get_shape
from tests.common.test_op.sign import sign


def _compute_m_t(m, beta, grad):
    """Update m."""
    beta_tmp = tvm.compute(m.shape,
                           lambda *indice: m(*indice) * beta[0])
    beta_na = tvm.compute(beta.shape,
                          lambda *indice: beta(*indice) * neg_one_const("float32"))
    beta_na = tvm.compute(beta_na.shape,
                          lambda *indice: beta_na(*indice) + one_const("float32"))
    beta_sub_tmp = tvm.compute(grad.shape,
                               lambda *indice: grad(*indice) * beta_na[0])
    m_t = topi.add(beta_tmp, beta_sub_tmp)
    return m_t


def _compute_update(logbase, sign_decay, sign_gm, grad):
    """Calculate var decay."""
    vmul_tmp = tvm.compute(sign_gm.shape,
                           lambda *indice: sign_gm(*indice) * sign_decay[0])
    vmul_tmp = tvm.compute(vmul_tmp.shape,
                           lambda *indice: vmul_tmp(*indice) * logbase[0])
    exp_tmp = exp(vmul_tmp)
    update = topi.multiply(exp_tmp, grad)
    return update


def _compute_var(var, lr, update):
    """Update var."""
    lt_tmp = tvm.compute(update.shape,
                         lambda *indice: update(*indice) * lr[0])
    var_t = topi.subtract(var, lt_tmp)
    return var_t


def _compute_process(var, m, grad, lr, logbase, sign_decay, beta):
    """Compute process of power_sign."""
    # m_t = beta * m + (1 - beta) * grad
    m_t = _compute_m_t(m, beta, grad)

    # update = exp(logbase * sign_decay * sign(m_t) * sign(grad)) * grad
    sign_gm = topi.multiply(sign(m_t), sign(grad))
    update = _compute_update(logbase, sign_decay, sign_gm, grad)
    # var_t = var - lr_t * update
    var_t = _compute_var(var, lr, update)

    return var_t, m_t


def _apply_power_sign_compute(var, m, grad, lr, logbase, sign_decay, beta):
    """Calculate the algorithm."""
    dtype = var.dtype
    if dtype == "float16":
        var = topi.cast(var, "float32")
        m = topi.cast(m, "float32")
        lr = topi.cast(lr, "float32")
        logbase = topi.cast(logbase, "float32")
        sign_decay = topi.cast(sign_decay, "float32")
        beta = topi.cast(beta, "float32")
        grad = topi.cast(grad, "float32")

    var_t, m_t = _compute_process(var, m, grad, lr, logbase, sign_decay, beta)

    if dtype == "float16":
        var_t = topi.cast(var_t, "float16")
        m_t = topi.cast(m_t, "float16")

    return var_t, m_t


@vc_util.check_input_type(*([akg.tvm.tensor.Tensor] * 7))
def apply_power_sign(var, m, grad, lr, logbase, sign_decay, beta):
    """
    Update 'var' according to the PowerSign update

    m_out = beta * m + (1 - beta) * grad
    var_out = var - lr_t * (exp(logbase * sign_decay * sign(grad) * sign(m_out)) * grad)

    Args:
        var (tvm.tensor.Tensor): A tensor of type float16 or float32
        m (tvm.tensor.Tensor): A tensor of same shape and type as var.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.
        lr (tvm.tensor.Tensor): A scalar tensor of of same type as var.
        logbase (tvm.tensor.Tensor): A scalar tensor of of same type as var.
        sign_decay (tvm.tensor.Tensor): A scalar tensor of of same type as var.
        beta (tvm.tensor.Tensor): A scalar tensor of of same type as var.

    Returns:
        tvm.tensor.Tensor, updated var.
        tvm.tensor.Tensor, updated m.
    """
    # check dtypes
    vc_util.ops_dtype_check(var.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    for i in (m, grad, lr, logbase, sign_decay, beta):
        vc_util.elemwise_dtype_check(var.dtype, i.dtype)

    # check shapes
    for i in (m, grad):
        vc_util.elemwise_shape_check(var.shape, i.shape)
    for i in (lr, logbase, sign_decay, beta):
        if tuple(get_shape(i)) != (1,):
            raise RuntimeError("lr, logbase, sign_decay and beta only support scalar tensor.")

    # compute
    out_var, out_m = _apply_power_sign_compute(var, m, grad, lr, logbase, sign_decay, beta)

    # reuse var, m
    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    out_m, binds_info2 = TensorUtils.inplace_set(m, out_m, "m_buf")
    binds_info.update(binds_info2)
    attrs = {utils.BINDS: binds_info}
    return out_var, out_m, attrs
