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

"""operator dsl fuction: apply_adadelta"""

import akg
from akg import topi, tvm
from akg.ops.math import Sqrt, rsqrt
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils
from akg.utils.format_transform import get_shape

def _apply_adadelta_compute(var, accum, accum_update, grad, lr, rho, epsilon):
    """Compute apply_adadelta"""
    dtype = var.dtype
    if dtype == "float16":
        var = topi.cast(var, "float32")
        accum = topi.cast(accum, "float32")
        accum_update = topi.cast(accum_update, "float32")
        lr = topi.cast(lr, "float32")
        rho = topi.cast(rho, "float32")
        grad = topi.cast(grad, "float32")

    epsilon = tvm.const(epsilon, "float32")
    tensor_one = akg.lang.ascend.broadcast(tvm.const(1, "float32"), var.shape)
    tensor_rho = topi.broadcast_to(rho, var.shape)
    tensor_rho_gs = topi.subtract(tensor_one, tensor_rho)
    tensor_epsilon = akg.lang.ascend.broadcast(epsilon, var.shape)

    # accum = accum * rho + grad ** 2 * (1 - rho)
    rhs = topi.multiply(accum, tensor_rho)
    lhs = topi.multiply(grad, grad)
    lhs = topi.multiply(lhs, tensor_rho_gs)
    accum_res = akg.lang.ascend.vadd(lhs, rhs)

    # update = (accum_update + epsilon).sqrt * (accum + epsilon).rsqrt * grad
    rhs = topi.add(accum_update, tensor_epsilon)
    rhs = Sqrt(rhs, target=utils.CCE)
    lhs = topi.add(accum_res, tensor_epsilon)
    lhs = rsqrt(lhs, target=utils.CCE)
    lhs = topi.multiply(grad, lhs)
    update = topi.multiply(lhs, rhs)

    # var -= update * lr
    var_res = topi.broadcast_to(lr, var.shape)
    var_res = topi.multiply(update, var_res)
    var_res = topi.subtract(var, var_res)

    # accum_update = rho * accum_update + (1 - rho) * update.square
    rhs = topi.multiply(accum_update, tensor_rho)
    lhs = topi.multiply(update, update)
    lhs = topi.multiply(lhs, tensor_rho_gs)
    accum_update_res = akg.lang.ascend.vadd(lhs, rhs)

    if dtype == "float16":
        var_res = topi.cast(var_res, "float16")
        accum_res = topi.cast(accum_res, "float16")
        accum_update_res = topi.cast(accum_update_res, "float16")

    return var_res, accum_res, accum_update_res


def _check_inputs(var, accum, accum_update, grad, lr, rho, epsilon):
    """Check op inputs"""
    # check dtype
    utils.ops_dtype_check(var.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    for i in (accum, accum_update, grad, lr, rho):
        utils.elemwise_dtype_check(var.dtype, i.dtype)

    # check shape
    for i in (accum, accum_update, grad):
        utils.elemwise_shape_check(var.shape, i.shape)
    for i in (lr, rho):
        if tuple(get_shape(i)) != (1,):
            raise RuntimeError("lr and rho only support scalar tensor.")

    # check value
    if epsilon <= 0:
        raise ValueError("epsilon should be greater than zero.")


@utils.check_input_type(*([akg.tvm.tensor.Tensor] * 6), float, (str, type(None)))
def apply_adadelta(var, accum, accum_update, grad, lr, rho, epsilon, target=utils.CCE):
    """
    Update var according to the adadelta scheme.

    accum = rho * accum + (1 - rho) * grad^2
    update = sqrt(accum_update + epsilon).sqrt() / sqrt(accum + epsilon) * grad
    accum_update = rho * accum_update + (1 - rho) * update^2
    var -= update * lr

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float32.
        accum (tvm.tensor.Tensor): The accumulate gradient, a tensor of same shape and type as var.
        accum_update (tvm.tensor.Tensor): The accumulate updates, tensor of same shape and type as var.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.
        lr (tvm.tensor.Tensor): Learning rate, a scalar tensor of same type as var.
        rho (tvm.tensor.Tensor): Coefficient for calculate new accum, 0.0 <= rho <= 1.0.
        epsilon (float): A small value to prevent division by 0.

    Returns:
        tvm.tensor.Tensor, Updated var.
        tvm.tensor.Tensor, Updated accum.
        tvm.tensor.Tensor, Updated accum_update.
    """

    _check_inputs(var, accum, accum_update, grad, lr, rho, epsilon)

    out_var, out_accum, out_accum_update = _apply_adadelta_compute(var, accum, accum_update, grad, lr, rho, epsilon)

    # reuse var, accum and accum_update
    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    out_accum, binds_info2 = TensorUtils.inplace_set(accum, out_accum, "accum_buf")
    out_accum_update, binds_info3 = TensorUtils.inplace_set(accum_update, out_accum_update, "accum_update_buf")
    binds_info.update(binds_info2)
    binds_info.update(binds_info3)
    attrs = {utils.BINDS: binds_info}
    return out_var, out_accum, out_accum_update, attrs
