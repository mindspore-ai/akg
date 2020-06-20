# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function: apply_rms_prop"""

import akg.tvm
from akg import topi
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.utils.dsl_create import TensorUtils
from akg.ops.math.rsqrt import rsqrt


def _apply_rms_prop_compute(var, ms, mom, grad, lr, momentum, rho, epsilon):
    """Compute apply_rms_prop"""
    shape = get_shape(var)
    dtype = var.dtype
    cons_eps = akg.tvm.const(epsilon, dtype=dtype)
    one_minus_rho = akg.tvm.compute((1, ), lambda *indice: akg.tvm.const(1.0, dtype) - rho[0], name="one_minus_rho")

    # var_update = var - (momentum * mom + lr * grad / sqrt(rho * ms + (1 - rho) * grad * grad + epsilon))
    mom_1 = akg.tvm.compute(shape, lambda *indice: momentum[0] * mom(*indice), name="mom_1")
    lr_grad = akg.tvm.compute(shape, lambda *indice: grad(*indice) * lr[0], name="lr_grad")
    rho_ms = akg.tvm.compute(shape, lambda *indice: ms(*indice) * rho[0], name="rho_ms")
    rho_grad2 = akg.tvm.compute(shape, lambda *indice: grad(*indice) *
                                grad(*indice) * one_minus_rho[0], name="rho_grad2")
    ms_update = akg.tvm.compute(shape, lambda *indice: rho_ms(*indice) + rho_grad2(*indice), name="ms_update")
    ms_eps = akg.tvm.compute(shape, lambda *indice: ms_update(*indice) + cons_eps, name="ms_eps")
    rsq = rsqrt(ms_eps)
    mom_2 = akg.tvm.compute(shape, lambda *indice: lr_grad(*indice) * rsq(*indice), name="mom_2")
    mom_update = akg.tvm.compute(shape, lambda *indice: mom_1(*indice) + mom_2(*indice), name="mom_update")
    var_update = akg.tvm.compute(shape, lambda *indice: var(*indice) - mom_update(*indice), name="var_update")

    return var_update, ms_update, mom_update


def _apply_rms_prop_mixed_precision_compute(var, ms, mom, grad, lr, momentum, rho, epsilon):
    """Compute apply_rms_prop_mixed_precision"""
    out_var, out_ms, out_mom = _apply_rms_prop_compute(var, ms, mom, grad, lr, momentum, rho, epsilon)
    out_var_fp16 = topi.cast(out_var, "float16")
    return out_var, out_var_fp16, out_ms, out_mom


def _apply_rms_prop_check(var, ms, mom, grad, lr, momentum, rho, epsilon):
    """Check inputs"""
    vc_util.check_shape(var)
    for i in (ms, mom, grad, lr, momentum, rho):
        vc_util.elemwise_dtype_check(var.dtype, i.dtype)
    for i in (ms, mom, grad):
        vc_util.elemwise_shape_check(var.shape, i.shape)
    for i in (lr, rho, momentum):
        if tuple(get_shape(i)) != (1,):
            raise RuntimeError("lr, rho and momentum only support scalar tensor.")
    if epsilon <= 0:
        raise ValueError("epsilon should greater than zero.")


@vc_util.check_input_type(*([akg.tvm.tensor.Tensor] * 7), float)
def apply_rms_prop(var, ms, mom, grad, lr, momentum, rho, epsilon):
    """
    Updates var using the RMSProp algorithm.

    .. math::
        \\begin{array}{ll} \\\\
            \\hat{ms} &= rho \\cdot ms + (1 - rho) \\cdot grad^2 \\\\
            \\hat{mom} &= momentum \\cdot mom +
                \\frac{lr \\cdot grad}{\\sqrt{\\hat{ms} + epsilon}} \\\\
            var &= var - mom
        \\end{array}

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float16 or float32.
        ms (tvm.tensor.Tensor): Mean square, a tensor of same shape and type as var.
        mom (tvm.tensor.Tensor): A tensor of same shape and type as var.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.
        lr (tvm.tensor.Tensor): Learning rate, a scalar tensor of same type as var.
        momentum (tvm.tensor.Tensor): Coefficient for calculate new mom, 0.0 <= momentum <= 1.0.
        rho (tvm.tensor.Tensor): Coefficient for calculate new ms, 0.0 <= rho <= 1.0.
        epsilon (float): A small value to prevent division by 0.

    Returns:
        tvm.tensor.Tensor, Updated var.
        tvm.tensor.Tensor, Updated ms.
        tvm.tensor.Tensor, Updated mom.
    """

    vc_util.ops_dtype_check(var.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    _apply_rms_prop_check(var, ms, mom, grad, lr, momentum, rho, epsilon)

    out_var, out_ms, out_mom = _apply_rms_prop_compute(var, ms, mom, grad, lr, momentum, rho, epsilon)
    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    out_ms, binds_info2 = TensorUtils.inplace_set(ms, out_ms, "ms_buf")
    out_mom, binds_info3 = TensorUtils.inplace_set(mom, out_mom, "mom_buf")
    binds_info.update(binds_info2)
    binds_info.update(binds_info3)
    attrs = {utils.BINDS: binds_info}
    return out_var, out_ms, out_mom, attrs


@vc_util.check_input_type(*([akg.tvm.tensor.Tensor] * 7), float)
def apply_rms_prop_mixed_precision(var, ms, mom, grad, lr, momentum, rho, epsilon):
    """
    Mixed precision version for apply_rms_prop.

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float32.
        ms (tvm.tensor.Tensor): Mean square, a tensor of same shape and type as var.
        mom (tvm.tensor.Tensor): A tensor of same shape and type as var.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.
        lr (tvm.tensor.Tensor): Learning rate, a scalar tensor of same type as var.
        momentum (float): Coefficient for calculate new mom, 0.0 <= momentum <= 1.0.
        rho (float): Coefficient for calculate new ms, 0.0 <= rho <= 1.0.
        epsilon (float): A small value to prevent division by 0.

    Returns:
        tvm.tensor.Tensor, Updated var of type float32.
        tvm.tensor.Tensor, Updated var of type float16.
        tvm.tensor.Tensor, Updated ms.
        tvm.tensor.Tensor, Updated mom.
    """

    vc_util.ops_dtype_check(var.dtype, vc_util.DtypeForDavinci.FLOAT32)
    _apply_rms_prop_check(var, ms, mom, grad, lr, momentum, rho, epsilon)

    out_var, out_var_fp16, out_ms, out_mom = _apply_rms_prop_mixed_precision_compute(
        var, ms, mom, grad, lr, momentum, rho, epsilon)
    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    out_ms, binds_info2 = TensorUtils.inplace_set(ms, out_ms, "ms_buf")
    out_mom, binds_info3 = TensorUtils.inplace_set(mom, out_mom, "mom_buf")
    binds_info.update(binds_info2)
    binds_info.update(binds_info3)
    attrs = {utils.BINDS: binds_info}
    return out_var, out_var_fp16, out_ms, out_mom, attrs
