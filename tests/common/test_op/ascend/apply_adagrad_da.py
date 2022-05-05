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

"""operator dsl function: apply_adagrad_da"""
import akg
import akg.utils as utils
from akg import topi, tvm
from akg.ops.math import Sqrt, reciprocal
from akg.ops.math.ascend import Sign
from akg.utils.kernel_exec import product_is_mini
from akg.utils.dsl_create import TensorUtils
from akg.utils.format_transform import get_shape

def _apply_adagrad_da_compute(var, gradient_accum, gradient_squared_accum,
                              grad, lr, l1, l2, global_step):
    """Compute adagrad_da."""
    dtype = var.dtype
    # cast to float32 for higher precision
    if dtype == "float16":
        gradient_accum = topi.cast(gradient_accum, "float32")
        gradient_squared_accum = topi.cast(gradient_squared_accum, "float32")
        grad = topi.cast(grad, "float32")
        lr = topi.cast(lr, "float32")
        l1 = topi.cast(l1, "float32")
        l2 = topi.cast(l2, "float32")
    if product_is_mini():
        global_step = topi.cast(global_step, "float16")
        global_step = topi.cast(global_step, "float32")
    else:
        global_step = topi.cast(global_step, "float32")

    # 1.grad_accum += grad
    gradient_accum = topi.add(gradient_accum, grad)

    # 2.grad_squared_accum += grad * grad
    gs = topi.multiply(grad, grad)
    gradient_squared_accum = topi.add(gradient_squared_accum, gs)

    # 3.if l1 > 0: tmp_val = Sign(grad_accum) * max(|grad_accum|-l1*global_step, 0)
    #   else:      tmp_val = grad_accum
    sign_val = Sign(gradient_accum)
    abs_val = topi.abs(gradient_accum)
    mul_val = topi.multiply(global_step, l1)
    sub_val = topi.subtract(abs_val, mul_val)
    max_val = topi.maximum(sub_val, tvm.const(0, sub_val.dtype))
    tmp_val = topi.multiply(sign_val, max_val)

    def select(l1, tmp_val, gradient_accum):
        """Returns tmp_val if l1 > 0 else gradient_accum."""
        if product_is_mini():
            l1 = topi.cast(l1, "float16")
            tmp_val = topi.cast(tmp_val, "float16")
            gradient_accum = topi.cast(gradient_accum, "float16")
        tmp_val = akg.tvm.compute(tmp_val.shape, lambda *i: tvm.expr.Select(
            l1[0] > 0, tmp_val(*i), gradient_accum(*i)))
        return topi.cast(tmp_val, "float32") if product_is_mini() else tmp_val
    tmp_val = select(l1, tmp_val, gradient_accum)

    # 4.x_value = -1 * lr * tmp_val
    x_value = topi.multiply(lr, tvm.const(-1, "float32"))
    x_value = topi.multiply(x_value, tmp_val)

    # 5.y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    pro_val = topi.multiply(l2, global_step)
    pro_val = topi.multiply(pro_val, lr)
    sqrt_val = Sqrt(gradient_squared_accum, target=utils.CCE)
    y_value = topi.add(pro_val, sqrt_val)

    # 6.var = x_value / y_value
    if product_is_mini():
        y_rec = reciprocal(y_value, target=utils.CCE)
        var_out = topi.multiply(x_value, y_rec)
    else:
        var_out = topi.divide(x_value, y_value)

    if dtype == "float16":
        var_out = akg.lang.ascend.cast_to(var_out, "float16")
        gradient_accum = akg.lang.ascend.cast_to(gradient_accum, "float16")
        gradient_squared_accum = akg.lang.ascend.cast_to(gradient_squared_accum, "float16")

    return var_out, gradient_accum, gradient_squared_accum


def _check_inputs(var, grad_accum, grad_squared_accum, grad, lr, l1, l2, global_step):
    """Check op inputs"""
    # check dtype
    utils.ops_dtype_check(var.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    for i in (grad_accum, grad_squared_accum, grad, lr, l1, l2):
        utils.elemwise_dtype_check(var.dtype, i.dtype)
    utils.ops_dtype_check(global_step.dtype, utils.DtypeForDavinci.INT32)

    # check shape
    for i in (grad_accum, grad_squared_accum, grad):
        utils.elemwise_shape_check(var.shape, i.shape)
    for i in (lr, l1, l2, global_step):
        if tuple(get_shape(i)) != (1,):
            raise RuntimeError("lr, l1, l2 and global_step only support scalar tensor.")


@utils.check_input_type(*([akg.tvm.tensor.Tensor] * 8), (str, type(None)))
def apply_adagrad_da(var, grad_accum, grad_squared_accum, grad, lr, l1, l2, global_step, target=utils.CCE):
    """
    Update var according to the Adagrad Dual Averaging algorithm.

    grad_accum += grad
    grad_squared_accum += grad * grad
    tmp_val = Sign(grad_accum) * max(|grad_accum|-l1*global_step, 0) if l1 > 0 else grad_accum
    x_value = -1 * lr * tmp_val
    y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    var = x_value / y_value

    Args:
        var (tvm.tensor.Tensor): Input var to be updated of type float16, float32.
        grad_accum (tvm.tensor.Tensor): Accumulation of the gradients of same shape and type as var.
        grad_squared_accum (tvm.tensor.Tensor): Accumulation of the squared gradients of same shape and type as var.
        grad (tvm.tensor.Tensor): Input grad of same shape and type as var.
        lr (tvm.tensor.Tensor): Learning rate, a scalar tensor of same type as var.
        l1 (tvm.tensor.Tensor): L1 regularization, a scalar tensor of same type as var.
        l2 (tvm.tensor.Tensor): L2 regularization, a scalar tensor of same type as var.
        global_step (tvm.tensor.Tensor): Training step number, a scalar tensor of type int32.

    Returns:
        tvm.tensor.Tensor, the updated var.
        tvm.tensor.Tensor, the updated grad_accum.
        tvm.tensor.Tensor, the updated grad_squared_accum.
    """

    _check_inputs(var, grad_accum, grad_squared_accum, grad, lr, l1, l2, global_step)

    out_var, out_ga, out_gsa = _apply_adagrad_da_compute(
        var, grad_accum, grad_squared_accum, grad, lr, l1, l2, global_step)

    # reuse var, grad_accum and grad_squared_accum
    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    out_ga, binds_info2 = TensorUtils.inplace_set(grad_accum, out_ga, "grad_accum_buf")
    out_gsa, binds_info3 = TensorUtils.inplace_set(grad_squared_accum, out_gsa, "grad_squared_accum_buf")
    binds_info.update(binds_info2)
    binds_info.update(binds_info3)
    attrs = {utils.BINDS: binds_info}
    return out_var, out_ga, out_gsa, attrs
