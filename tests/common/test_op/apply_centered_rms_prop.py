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

"""operator dsl function: apply_centered_rms_prop"""
import akg
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import TensorUtils
from akg.utils.format_transform import get_shape
from akg.ops.math.rsqrt import rsqrt


def _apply_centered_rms_prop_compute(var, mg, ms, mom, grad, lr, momentum, rho, epsilon):
    """Compute apply_centered_rms_prop"""
    inp_dtype = var.dtype
    if inp_dtype == "float16":
        var = akg.lang.cce.cast_to(var, "float32")
        mg = akg.lang.cce.cast_to(mg, "float32")
        ms = akg.lang.cce.cast_to(ms, "float32")
        mom = akg.lang.cce.cast_to(mom, "float32")
        lr = akg.lang.cce.cast_to(lr, "float32")
        rho = akg.lang.cce.cast_to(rho, "float32")
        momentum = akg.lang.cce.cast_to(momentum, "float32")
        grad = akg.lang.cce.cast_to(grad, "float32")
    epsilon = akg.tvm.const(epsilon, var.dtype)

    tensor_one_rho = akg.tvm.compute(rho.shape,
                                     lambda *indices: rho(*indices) * akg.tvm.const(-1, rho.dtype),
                                     tag='elewise_single_VS_mul')
    tensor_one_rho = akg.tvm.compute(
        tensor_one_rho.shape,
        lambda *indices: tensor_one_rho(*indices) + akg.tvm.const(1, rho.dtype),
        tag='elewise_single_VS_add')

    # out_mg <- rho * mg + (1-rho) * grad
    mg_rho = akg.tvm.compute(mg.shape,
                             lambda *indices: mg(*indices) * rho[0],
                             tag='elewise_single_VS_mul')
    rhs = akg.tvm.compute(grad.shape,
                          lambda *indices: grad(*indices) * tensor_one_rho[0],
                          tag='elewise_single_VS_mul')
    out_mg = akg.lang.cce.vadd(mg_rho, rhs)

    # out_ms <- rho * ms + (1-rho) * grad * grad
    ms_rho = akg.tvm.compute(ms.shape,
                             lambda *indices: ms(*indices) * rho[0],
                             tag='elewise_single_VS_mul')
    rhs = akg.lang.cce.vmul(grad, grad)
    rhs = akg.tvm.compute(rhs.shape,
                          lambda *indices: rhs(*indices) * tensor_one_rho[0],
                          tag='elewise_single_VS_mul')
    out_ms = akg.lang.cce.vadd(ms_rho, rhs)

    # out_mom <- momentum * mom + lr * grad / sqrt(out_ms - out_mg * out_mg + epsilon)
    lhs_mom = akg.tvm.compute(mom.shape,
                              lambda *indices: mom(*indices) * momentum[0],
                              tag='elewise_single_VS_mul')
    lr_grad = akg.tvm.compute(grad.shape,
                              lambda *indices: grad(*indices) * lr[0],
                              tag='elewise_single_VS_mul')
    rhs = akg.lang.cce.vmul(out_mg, out_mg)
    rhs = akg.lang.cce.vsub(out_ms, rhs)
    rhs_eps = akg.tvm.compute(rhs.shape,
                              lambda *indices: rhs(*indices) + epsilon,
                              tag='elewise_single_VS_add')
    rhs_eps = rsqrt(rhs_eps)
    rhs_eps = akg.lang.cce.vmul(lr_grad, rhs_eps)
    out_mom = akg.lang.cce.vadd(lhs_mom, rhs_eps)

    # out_var <- var - out_mom
    out_var = akg.lang.cce.vsub(var, out_mom)

    if inp_dtype == "float16":
        out_var = akg.lang.cce.cast_to(out_var, "float16")
        out_mg = akg.lang.cce.cast_to(out_mg, "float16")
        out_ms = akg.lang.cce.cast_to(out_ms, "float16")
        out_mom = akg.lang.cce.cast_to(out_mom, "float16")

    return out_var, out_mg, out_ms, out_mom


@vc_util.check_input_type(*([akg.tvm.tensor.Tensor] * 8), float)
def apply_centered_rms_prop(var, mg, ms, mom, grad, lr, momentum, rho, epsilon):
    """
    Update `var` according to the centered RMSProp algorithm.

    out_mean_grad = decay * mg + (1-decay) * grad
    out_mean_square = decay * ms + (1-decay) * grad * grad
    out_mom = momentum * mom + lr * grad / sqrt(out_mean_square - out_mean_grad^2 + epsilon)
    out_var = var - out_mom

    Args:
        var (tvm.tensor.Tensor): Input data of type float16 or float32.
        mg (tvm.tensor.Tensor): A tensor of the same type and shape as `var`.
        ms (tvm.tensor.Tensor): A tensor of the same type and shape as `var`.
        mom (tvm.tensor.Tensor): A tensor of the same type and shape as `var`.
        grad (tvm.tensor.Tensor): A tensor of the same type and shape as `var`.
        lr (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        momentum (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        rho (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        epsilon (float): A scalar tensor of the same type as `var`.

    Returns:
        tvm.tensor.Tensor, updated var.
        tvm.tensor.Tensor, updated mean_grad.
        tvm.tensor.Tensor, updated mean_square.
        tvm.tensor.Tensor, updated mom.
    """

    vc_util.ops_dtype_check(var.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    for i in (mg, ms, mom, lr, rho, momentum, grad):
        vc_util.elemwise_dtype_check(var.dtype, i.dtype)
    for i in (mg, ms, mom, grad):
        vc_util.elemwise_shape_check(var.shape, i.shape)
    for i in (lr, rho, momentum):
        if tuple(get_shape(i)) != (1,):
            raise RuntimeError("lr, rho and momentum only support scalar tensor.")
    if epsilon <= 0:
        raise ValueError("epsilon should be greater than 0.")

    out_var, out_mg, out_ms, out_mom = _apply_centered_rms_prop_compute(
        var, mg, ms, mom, grad, lr, momentum, rho, epsilon)
    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    out_mg, binds_info2 = TensorUtils.inplace_set(mg, out_mg, "mg_buf")
    out_ms, binds_info3 = TensorUtils.inplace_set(ms, out_ms, "ms_buf")
    out_mom, binds_info4 = TensorUtils.inplace_set(mom, out_mom, "mom_buf")
    binds_info.update(binds_info2)
    binds_info.update(binds_info3)
    binds_info.update(binds_info4)
    attrs = {utils.BINDS: binds_info}
    return out_var, out_mg, out_ms, out_mom, attrs
