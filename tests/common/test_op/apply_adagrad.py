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

"""operator dsl function: apply_adagrad"""
import akg
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape

from akg.ops.math.rsqrt import rsqrt


def _apply_adagrad_compute(var, accum, lr, grad, update_slots):
    """Compute apply_adagrad"""
    input_dtype = var.dtype
    if input_dtype == "float16":
        var = akg.lang.cce.cast_to(var, "float32")
        accum = akg.lang.cce.cast_to(accum, "float32")
        lr = akg.lang.cce.cast_to(lr, "float32")
        grad = akg.lang.cce.cast_to(grad, "float32")

    if update_slots is True:
        # accum += grad ** 2
        grad_square = akg.lang.cce.vmul(grad, grad)
        accum = akg.lang.cce.vadd(accum, grad_square)
    elif input_dtype == 'float32':
        accum = akg.lang.cce.vadds(accum, akg.tvm.const(0, "float32"))

    # var -= lr * grad / accum.sqrt()
    lr_grad = akg.tvm.compute(grad.shape,
                              lambda *indices: grad(*indices) * lr[0],
                              tag='elewise_single_VS_mul')
    rsqrt_accum = rsqrt(accum)

    update = akg.lang.cce.vmul(lr_grad, rsqrt_accum)
    out_var = akg.lang.cce.vsub(var, update)

    if input_dtype == "float16":
        out_var = akg.lang.cce.cast_to(out_var, "float16")
        accum = akg.lang.cce.cast_to(accum, "float16")

    return out_var, accum


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          (bool, type(None)))
def apply_adagrad(var, accum, learning_rate, grad, update_slots=True):
    """
    Update `var` according to the Adagrad algorithm.

    .. math:
        accum += grad^2
        var -= learning_rate * grad / accum.sqrt()

    Args:
        var (tvm.tensor.Tensor): input var to be updated of type float16, float32
        accum (tvm.tensor.Tensor): accumulation of the squared gradients of type float16, float32
        learning_rate (tvm.tensor.Tensor): A scalar tensor of type float16, float32
        grad (tvm.tensor.Tensor): input grad of type float16, float32
        update_slots (Bool): If True, the accum tensor will be updated;
            otherwise the option is False, the accum tensor will not be update.
            Defaults to 'True'.

    Returns:
        tvm.tensor.Tensor, the updated var.
        tvm.tensor.Tensor, the updated accum.
    """

    vc_util.ops_dtype_check(var.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    for i in (accum, learning_rate, grad):
        vc_util.elemwise_dtype_check(var.dtype, i.dtype)
    for i in (accum, grad):
        vc_util.elemwise_shape_check(var.shape, i.shape)
    if tuple(get_shape(learning_rate)) != (1,):
        raise RuntimeError("learning_rate only support scalar tensor")

    return _apply_adagrad_compute(var, accum, learning_rate, grad, update_slots)
