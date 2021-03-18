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

"""operator dsl function: apply_ftrl"""

import akg.tvm
import akg.topi
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils
from akg.ops.math.div import div


def apply_ftrl_impl(var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power, with_l2_shrinkage=False):
    """Ftrl-proximal Optimization algorithm"""

    dtype = var.dtype
    # cast to float32 for higher accuracy
    compute_dtype = dtype
    if dtype == "float16":
        compute_dtype = "float32"
        var, accum, linear, grad, lr, l1, l2, lr_power = [akg.topi.cast(t, compute_dtype) for t in
                                                          [var, accum, linear, grad, lr, l1, l2, lr_power]]
        if with_l2_shrinkage:
            l2_shrinkage = akg.topi.cast(l2_shrinkage, compute_dtype)

    shape = var.shape
    # grad_shrinkage = grad + 2 * l2_shrinkage * var
    if with_l2_shrinkage:
        l2_shrinkage = akg.topi.broadcast_to(l2_shrinkage, shape)
        grad_shrinkage = akg.tvm.compute(shape, lambda *indice:
                                         grad(*indice) + akg.tvm.const(2.0, compute_dtype) * l2_shrinkage(*indice) *
                                         var(*indice), name="grad_shrinkage")
    else:
        grad_shrinkage = grad

    # accum_new = accum + grad^2
    accum_new = akg.tvm.compute(shape, lambda *indice: accum(*indice) + grad(*indice)*grad(*indice), name="accum_new")

    # linear_new = linear +  grad_shrinkage - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
    lr_power_neg = akg.topi.negative(lr_power)
    accum_new_pow = akg.topi.power(accum_new, lr_power_neg)
    accum_pow = akg.topi.power(accum, lr_power_neg)
    accum_pow_sub = akg.topi.subtract(accum_new_pow, accum_pow)
    accum_pow_sub_div_lr = div(accum_pow_sub, lr)
    linear_add_shrinkage = akg.topi.add(linear, grad_shrinkage)
    linear_new = akg.tvm.compute(shape, lambda *indice:
                                 linear_add_shrinkage(*indice) - accum_pow_sub_div_lr(*indice)*var(*indice),
                                 name="linear_new")

    # x = clip(linear_new, -l1, l1) - linear_new
    l1_neg = akg.topi.negative(l1)
    linear_new_clip = akg.topi.minimum(akg.topi.maximum(linear_new, l1_neg), l1)
    x_res = akg.topi.subtract(linear_new_clip, linear_new)
    # y = accum_new^(-lr_power) / lr + 2 * l2
    accum_new_pow_div_lr = div(accum_new_pow, lr)
    l2_2 = akg.topi.multiply(l2, 2)
    y_res = akg.topi.add(accum_new_pow_div_lr, l2_2)
    # var_new = x / y
    var_new = div(x_res, y_res)

    # cast to original type
    if dtype == "float16":
        var_new = akg.topi.cast(var_new, dtype)
        accum_new = akg.topi.cast(accum_new, dtype)
        linear_new = akg.topi.cast(linear_new, dtype)

    return var_new, accum_new, linear_new


@vc_util.check_input_type(*([akg.tvm.tensor.Tensor]*8))
def apply_ftrl(var, accum, linear, grad, lr, l1, l2, lr_power):
    """
    Ftrl-proximal optimization algorithm.

    Note:
        accum_new = accum + grad * grad
        linear_new = linear +  grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
        x = clip(linear_new, -l1, l1) - linear_new
        y = accum_new^(-lr_power) / lr + 2 * l2
        var_new = x / y

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float16 or float32.
        accum (tvm.tensor.Tensor): A tensor of same shape and type as var. Eatch entry in it must be
                                   greater or equal to zero.
        linear (tvm.tensor.Tensor): A tensor of same shape and type as var.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.
        lr (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        l1 (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        l2 (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        lr_power (tvm.tensor.Tensor): A scalar tensor of the same type as `var`. Value of it
                                      must be less or equal to zero.

    Returns:
        tvm.tensor.Tensor, updated var.
        tvm.tensor.Tensor, updated accum.
        tvm.tensor.Tensor, updated linear.
    """

    # As vlog instruction on mini product has a percision problem and mini product used to infer
    # rather than train
    if utils.product_is_mini():
        raise RuntimeError("The apply_ftrl operator does not support the mini product")

    # check_shape
    vc_util.check_shape(var)
    shape = get_shape(var)
    for tensor in (accum, linear, grad):
        vc_util.elemwise_shape_check(shape, tensor.shape)
    sclar_shape = (1,)
    for sclar in (lr, l1, l2, lr_power):
        vc_util.elemwise_shape_check(sclar.shape, sclar_shape)

    # check dtype
    dtype = var.dtype
    vc_util.ops_dtype_check(dtype, [vc_util.DtypeForDavinci.FLOAT16, vc_util.DtypeForDavinci.FLOAT32])
    for tensor in (var, accum, linear, grad, lr, l1, l2, lr_power):
        vc_util.elemwise_dtype_check(tensor.dtype, dtype)

    var_new, accum_new, linear_new = apply_ftrl_impl(var, accum, linear, grad, lr, l1, l2, None,
                                                     lr_power, with_l2_shrinkage=False)

    # update by inplace
    (var_new, accum_new, linear_new), binds_info = \
        utils.TensorUtils.inplace_set_tensors((var, accum, linear), (var_new, accum_new, linear_new))
    attrs = {utils.BINDS: binds_info}
    return var_new, accum_new, linear_new, attrs
