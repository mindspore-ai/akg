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

"""operator dsl function: xdivy_grad"""
import akg
from akg import tvm
from akg.ops.math import Divide
from akg.ops.math import reciprocal
from akg.utils.format_transform import get_shape
from akg.utils.dsl_create import produce_shapes, broadcast_gradient_args
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini

def xdivy_grad_compute(placeholders, shape_max, dtype, rx, ry):
    """
    Do element-wise xdivy_grad compute.

    Args:
        placeholders (Union[list, typle]): the placeholder of data input
        shape_max (Union[list, typle]): the shape of broadcast
        dtype (string): the type of data input
        rx (list): the reduction indices of data input with broadcast
        ry (list): the reduction indices for data input with broadcast

    Returns:
        output_y1 (tvm.tensor.Tensor): result of xdivy_grad
        output_y2 (tvm.tensor.Tensor): result of xdivy_grad
    """
    x1_ori = placeholders[0]
    x2_ori = placeholders[1]
    grad_ori = placeholders[2]

    if dtype == "float16":
        x1 = akg.lang.ascend.cast_to(x1_ori, "float32")
        x2 = akg.lang.ascend.cast_to(x2_ori, "float32")
        grad = akg.lang.ascend.cast_to(grad_ori, "float32")
        x1 = akg.lang.ascend.broadcast(x1, shape_max)
        x2 = akg.lang.ascend.broadcast(x2, shape_max)
        grad = akg.lang.ascend.broadcast(grad, shape_max)
    else:
        x1 = akg.lang.ascend.broadcast(x1_ori, shape_max)
        x2 = akg.lang.ascend.broadcast(x2_ori, shape_max)
        grad = akg.lang.ascend.broadcast(grad_ori, shape_max)

    esp_min = tvm.const(1.18e-38, dtype="float32")
    x1_addepsmin = akg.lang.ascend.vadds(x1, esp_min)

    if product_is_mini():
        x1_addepsmin_rec = reciprocal(x1_addepsmin)
        not_zero_x1 = akg.lang.ascend.vmul(x1, x1_addepsmin_rec)
        x2_rec = reciprocal(x2)
        partial_x1 = akg.lang.ascend.vmul(not_zero_x1, x2_rec)
    else:
        not_zero_x1 = Divide(x1, x1_addepsmin, target="cce")
        partial_x1 = Divide(not_zero_x1, x2, target="cce")

    partial_x1g = akg.lang.ascend.vmul(partial_x1, grad)

    neg_one = tvm.const(-1, dtype="float32")
    neg_x1 = akg.lang.ascend.vmuls(x1, neg_one)
    partial_x1pow = akg.lang.ascend.vmul(partial_x1, partial_x1)
    partial_x2 = akg.lang.ascend.vmul(neg_x1, partial_x1pow)
    partial_x2g = akg.lang.ascend.vmul(partial_x2, grad)

    output_y1 = akg.lang.ascend.sum(partial_x1g, rx, keepdims=True)
    output_y2 = akg.lang.ascend.sum(partial_x2g, ry, keepdims=True)

    if dtype == "float16":
        output_y1 = akg.lang.ascend.cast_to(output_y1, "float16")
        output_y2 = akg.lang.ascend.cast_to(output_y2, "float16")

    return output_y1, output_y2

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, (str, type(None)))
def xdivy_grad(x1, x2, grad, target=utils.CCE):
    """
    Returns gradient of xdivy(x1, x2) with respect to x1 and x2.

    Args:
        x1 (tvm.tensor.Tensor): Tensor of dtype "float16" or "float32".
        x2 (tvm.tensor.Tensor): Tensor of dtype "float16" or "float32".
        grad (tvm.tensor.Tensor): Gradient tensor of dtype "float16" or "float32".

    Returns:
        Two tvm.tensor.Tensor as gradients for x1 and x2.
    """
    shape_x1 = get_shape(x1)
    dtype_x1 = x1.dtype
    shape_x2 = get_shape(x2)
    dtype_x2 = x2.dtype
    shape_grad = get_shape(grad)
    dtype_grad = grad.dtype
    if dtype_x1 != dtype_x2 or dtype_x2 != dtype_grad or dtype_grad != dtype_x1:
        raise RuntimeError(
            "the type of x1, x2 and grad must be the same,"
            "while dtype_x1 = %s, dtype_x2 = %s, dtype_grad = %s" %
            (dtype_x1, dtype_x2, dtype_grad))

    utils.check_shape(shape_x1)
    utils.check_shape(shape_x2)
    utils.check_shape(shape_grad)

    utils.ops_dtype_check(dtype_x1, utils.DtypeForDavinci.ALL_FLOAT)
    shape_x1, shape_x2, shape_max_x1x2 = produce_shapes(shape_x1, shape_x2)
    if len(shape_max_x1x2) < len(shape_grad):
        raise RuntimeError(
            "the length of shape_grad can not be longer than the maximum "
            "length of x1 and x2, while shape_grad = %s, shape_max= %s" %
            (list(shape_grad), shape_max_x1x2))

    shape_grad, _, shape_max = produce_shapes(shape_grad, shape_max_x1x2)
    for (x, y) in zip(shape_max_x1x2, shape_grad):
        if x < y:
            raise RuntimeError(
                "Don't support this shape. while shape_max = %s, shape_grad "
                "= %s" % (shape_max_x1x2, list(shape_grad)))

    rx, ry = broadcast_gradient_args(shape_x1, shape_x2)
    return xdivy_grad_compute([x1, x2, grad], shape_max,
                              dtype_x1, rx, ry)
