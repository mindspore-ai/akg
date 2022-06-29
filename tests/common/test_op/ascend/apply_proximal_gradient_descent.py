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

"""operator dsl function: apply_proximal_gradient_descent"""


import akg.tvm
import akg.topi
from akg.utils.format_transform import get_shape
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils
from akg.ops.math import divide
from akg.utils.kernel_exec import product_is_mini
from akg.ops.math.ascend import Sign


def apply_proximal_gradient_descent_impl(var, alpha, l1, l2, delta):
    """implement the FOBOS algorithm"""

    dtype = var.dtype
    compute_type = dtype
    if dtype == "float16":
        # cast to float32 for higher accuracy
        compute_type = "float32"
        var, alpha, l1, l2, delta = [akg.topi.cast(t, compute_type) for t in [var, alpha, l1, l2, delta]]

    shape = var.shape
    alpha = akg.topi.broadcast_to(alpha, shape)
    l1 = akg.topi.broadcast_to(l1, shape)
    l2 = akg.topi.broadcast_to(l2, shape)
    # prox_var = var - alpha * delta
    prox_var = akg.tvm.compute(shape, lambda *indice: var(*indice) - alpha(*indice)*delta(*indice), name="prox_var")

    # l1>0: var_new = Sign(prox_var)/(1+alpha*l2) * max{|prox_var|-alpha*l1,0}
    sign_prox_var = Sign(prox_var)
    alpha_l2_1 = akg.topi.add(akg.tvm.const(1, compute_type), akg.topi.multiply(alpha, l2))
    max_value = akg.tvm.compute(shape, lambda *indice: akg.tvm.max(
        akg.tvm.abs(prox_var(*indice)) - alpha(*indice)*l1(*indice),
        akg.tvm.const(0, compute_type)), name="max_value")
    var_new_l1_gt_0 = akg.topi.multiply(divide(sign_prox_var, alpha_l2_1, target="cce"), max_value)

    # l1<=0: var_new = prox_var/(1+alpha*l2)
    var_new_l1_le_0 = divide(prox_var, alpha_l2_1, target="cce")

    if product_is_mini():
        var_new = akg.tvm.compute(shape, lambda *indice:
                                  akg.tvm.expr.Select(l1(*indice).astype("float16") > akg.tvm.const(0, "float16"),
                                                      var_new_l1_gt_0(*indice).astype("float16"),
                                                      var_new_l1_le_0(*indice).astype("float16")),
                                  name="var_new")
    else:
        var_new = akg.tvm.compute(shape, lambda *indice:
                                  akg.tvm.expr.Select(l1(*indice) > akg.tvm.const(0, l1.dtype),
                                                      var_new_l1_gt_0(*indice), var_new_l1_le_0(*indice)),
                                  name="var_new")

    # cast to origin dtype
    if var_new.dtype != dtype:
        var_new = akg.topi.cast(var_new, dtype)
    return var_new


@utils.check_input_type(*([akg.tvm.tensor.Tensor]*5), (str, type(None)))
def apply_proximal_gradient_descent(var, alpha, l1, l2, delta, target=utils.CCE):
    """
    The FOBOS algorithm with fixed learning rate.

    Note:
        prox_var = var - alpha * delta
        if l1 > 0:
            var_new = Sign(prox_var)/(1+alpha*l2) * max{|prox_var|-alpha*l1,0}
        else:
            var_new = prox_var/(1+alpha*l2)

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float16 or float32.
        alpha (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        l1 (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        l2 (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        delta (tvm.tensor.Tensor): A tensor of same shape and type as var.

    Returns:
        tvm.tensor.Tensor, updated var.
    """

    # check_shape
    utils.check_shape(var)
    shape = get_shape(var)
    utils.elemwise_shape_check(shape, delta.shape)
    sclar_shape = (1,)
    for sclar in (alpha, l1, l2):
        utils.elemwise_shape_check(sclar.shape, sclar_shape)

    # check dtype
    dtype = var.dtype
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.FLOAT16, utils.DtypeForDavinci.FLOAT32])
    for tensor in (var, alpha, l1, l2, delta):
        utils.elemwise_dtype_check(tensor.dtype, dtype)

    var_new = apply_proximal_gradient_descent_impl(var, alpha, l1, l2, delta)
    var_new, binds_info = TensorUtils.inplace_set(var, var_new, "var_buf")
    attrs = {utils.BINDS: binds_info}
    return var_new, attrs
