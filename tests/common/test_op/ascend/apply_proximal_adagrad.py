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

"""operator dsl function: apply_proximal_adagrad"""


import akg.tvm
import akg.topi
from akg.utils.format_transform import get_shape
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils
from akg.ops.math import Rsqrt
from tests.common.test_op.ascend.apply_proximal_gradient_descent import apply_proximal_gradient_descent_impl


def _apply_proximal_adagrad_compute(var, accum, lr, l1, l2, grad):
    """compute the FOBOS algorithm with adagrad learning rate"""

    dtype = var.dtype
    if dtype == "float16":
        # cast to float32 for higher accuracy
        compute_type = "float32"
        var, accum, lr, l1, l2, grad = [akg.topi.cast(t, compute_type) for t in [var, accum, lr, l1, l2, grad]]

    shape = var.shape
    accum_new = akg.tvm.compute(shape, lambda *indice: accum(*indice) + grad(*indice) * grad(*indice), name="accum_new")

    accum_new_rsqrt = Rsqrt(accum_new, target="cce")
    ada_lr = akg.topi.multiply(lr, accum_new_rsqrt)

    var_new = apply_proximal_gradient_descent_impl(var, ada_lr, l1, l2, grad)

    # cast to origin dtype
    var_new, accum_new = [akg.topi.cast(t, dtype) if t.dtype != dtype else t for t in [var_new, accum_new]]
    return var_new, accum_new


@utils.check_input_type(*([akg.tvm.tensor.Tensor]*6), (str, type(None)))
def apply_proximal_adagrad(var, accum, lr, l1, l2, grad, target=utils.CCE):
    """
    The FOBOS optimization algorithm with Adagrad learning rate.

    Note:
        accum_new = accum + grad * grad
        ada_lr = lr * rsqrt(accum_new)
        prox_var = var - ada_lr * grad
        if l1 > 0:
            var_new = Sign(prox_var)/(1+ada_lr*l2) * max{|prox_var|-ada_lr*l1,0}
        else:
            var_new = prox_var/(1+ada_lr*l2)

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float16 or float32.
        accum (tvm.tensor.Tensor): A tensor of same shape and type as var. Eatch entry in it must be
                                   greater or equal to zero.
        lr (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        l1 (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        l2 (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.

    Returns:
        tvm.tensor.Tensor, updated var.
        tvm.tensor.Tensor, updated accum.
    """

    # check_shape
    utils.check_shape(var)
    shape = get_shape(var)
    for tensor in (accum, grad):
        utils.elemwise_shape_check(shape, tensor.shape)
    sclar_shape = (1,)
    for sclar in (lr, l1, l2):
        utils.elemwise_shape_check(sclar.shape, sclar_shape)

    # check dtype
    dtype = var.dtype
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.FLOAT16, utils.DtypeForDavinci.FLOAT32])
    for tensor in (var, accum, lr, l1, l2, grad):
        utils.elemwise_dtype_check(tensor.dtype, dtype)

    var_new, accum_new = _apply_proximal_adagrad_compute(var, accum, lr, l1, l2, grad)
    (var_new, accum_new), binds_info = TensorUtils.inplace_set_tensors([var, accum], [var_new, accum_new])
    attrs = {utils.BINDS: binds_info}
    return var_new, accum_new, attrs
