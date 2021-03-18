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

"""operator dsl function: apply_add_sign"""
import akg
from akg import topi, tvm
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import TensorUtils
from akg.utils.format_transform import get_shape
from tests.common.test_op.sign import sign


def _apply_add_sign_compute(var, m, grad, lr, alpha, sign_decay, beta):
    """Compute apply_add_sign"""
    m_out = _update_m(m, beta, grad)
    sign_gm = topi.multiply(sign(grad), sign(m_out))
    decay_gm = topi.multiply(sign_gm, sign_decay)
    var_out = _update_var(decay_gm, alpha, lr, grad, var)
    return var_out, m_out


def _update_m(m, beta, grad):
    """Update m_out = m * beta + grad * (1 - beta)"""
    m_beta = topi.multiply(m, beta)
    beta_neg = topi.multiply(beta, tvm.const(-1, beta.dtype))
    beta_1 = topi.add(beta_neg, tvm.const(1, beta_neg.dtype))
    grad_beta_gs = topi.multiply(grad, beta_1)
    m_out = topi.add(m_beta, grad_beta_gs)
    return m_out


def _update_var(decay_gm, alpha, lr, grad, var):
    """Update var_out = var - lr * (alpha + decay_gm) * grad"""
    decay_gm_alpha = topi.add(decay_gm, alpha)
    res = topi.multiply(decay_gm_alpha, lr)
    res = topi.multiply(res, grad)
    res_neg = topi.multiply(res, tvm.const(-1, res.dtype))
    var_out = topi.add(var, res_neg)
    return var_out


@vc_util.check_input_type(*([tvm.tensor.Tensor] * 7))
def apply_add_sign(var, m, grad, lr, alpha, sign_decay, beta):
    """
    Update 'var' according to the AddSign update.

    m_out = m * beta + grad * (1 - beta)
    var_out = var - lr * (alpha + sign_decay * sign(grad) *sign(m)) * grad

    Args:
        var (tvm.tensor.Tensor): A tensor of type float16 or float32
        m (tvm.tensor.Tensor): A tensor of type float16 or float32
        grad (tvm.tensor.Tensor): A tensor of type float16 or float32
        lr (tvm.tensor.Tensor): A scalar tensor of type float16 or float32
        alpha (tvm.tensor.Tensor): A scalar tensor of type float16 or float32
        sign_decay (tvm.tensor.Tensor): A scalar tensor of type float16 or float32
        beta (tvm.tensor.Tensor): A scalar tensor of type float16 or float32

    Returns:
        tvm.tensor.Tensor, updated var.
        tvm.tensor.Tensor, updated m.
    """

    vc_util.ops_dtype_check(var.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    for i in (m, lr, alpha, sign_decay, beta, grad):
        vc_util.elemwise_dtype_check(var.dtype, i.dtype)
    for i in (m, grad):
        vc_util.elemwise_shape_check(var.shape, i.shape)
    for i in (lr, alpha, sign_decay, beta):
        if tuple(get_shape(i)) != (1,):
            raise RuntimeError("lr, alpha, sign_decay and beta only support scalar.")

    out_var, out_m = _apply_add_sign_compute(var, m, grad, lr, alpha, sign_decay, beta)

    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    out_m, binds_info2 = TensorUtils.inplace_set(m, out_m, "m_buf")
    binds_info.update(binds_info2)
    attrs = {utils.BINDS: binds_info}
    return out_var, out_m, attrs
