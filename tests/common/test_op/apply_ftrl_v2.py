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

"""operator dsl function: apply_ftrl_v2"""

import akg.topi
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils

from tests.common.test_op.apply_ftrl import apply_ftrl_impl



@vc_util.check_input_type(*([akg.tvm.tensor.Tensor]*9))
def apply_ftrl_v2(var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power):
    """
    Ftrl-proximal optimization algorithm with l2_shrinkage.

    Note:
        grad_shrinkage = grad + 2 * l2_shrinkage * var
        accum_new = accum + grad * grad
        linear_new = linear +  grad_shrinkage - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
        x = clip(linear_new, -l1, l1) - linear_new
        y = accum_new^(-lr_power) / lr + 2 * l2
        var_new = x / y

    Args:
        var (tvm.tensor.Tensor): The tensor to be updated. Should be float16 or float32.
        accum (tvm.tensor.Tensor): A tensor of same shape and type as var. Eatch entry in it must be
                                   greater or equal to zero.
        linear (tvm.tensor.Tensor): A tensor of same shape and type as var.
        grad (tvm.tensor.Tensor): A tensor of same shape and type as var.
        lr (tvm.tensor.Tensor):  A scalar tensor of the same type as `var`.
        l1 (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        l2 (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
        l2_shrinkage (tvm.tensor.Tensor): A scalar tensor of the same type as `var`.
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
        raise RuntimeError("The apply_ftrl_v2 operator does not support the mini product")

    # check_shape
    vc_util.check_shape(var)
    shape = get_shape(var)
    for tensor in (accum, linear, grad):
        vc_util.elemwise_shape_check(shape, tensor.shape)
    sclar_shape = (1,)
    for sclar in (lr, l1, l2, l2_shrinkage, lr_power):
        vc_util.elemwise_shape_check(sclar.shape, sclar_shape)

    # check dtype
    dtype = var.dtype
    vc_util.ops_dtype_check(dtype, [vc_util.DtypeForDavinci.FLOAT16, vc_util.DtypeForDavinci.FLOAT32])
    for tensor in (var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power):
        vc_util.elemwise_dtype_check(tensor.dtype, dtype)

    var_new, accum_new, linear_new = apply_ftrl_impl(var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power,
                                                     with_l2_shrinkage=True)

    # update by inplace
    (var_new, accum_new, linear_new), binds_info = utils.TensorUtils.\
        inplace_set_tensors((var, accum, linear), (var_new, accum_new, linear_new))
    attrs = {utils.BINDS: binds_info}
    return var_new, accum_new, linear_new, attrs
