# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: softmax_grad"""

import akg.tvm
import akg
from akg.utils import custom_tiling as ct_util
import akg.utils as utils
from akg.ops.nn.ascend import Softmax

softmax_grad_set_dim_map = {
    str((((8, 4718, 6), -1), "float16")): ((1, 1), (674, 1), (16, 1)),
}


def softmax_grad_set_dim_func(x, dy, axis):
    key = []
    shape = [int(x.shape[i].value) for i in range(len(x.shape))]
    key.append(tuple(shape))
    key.append(axis)
    dtype = x.dtype

    hash_key = str((tuple(key), dtype))
    return ct_util.set_dims_by_key(hash_key, softmax_grad_set_dim_map), hash_key


@ct_util.reg_set_dim_func(softmax_grad_set_dim_func)
def softmax_grad(x, dy, axis):
    """
    Computes gradients of softmax.

    Args:
        x(akg.tvm.Tensor): Tensor of type float16 or float32.
        dy(akg.tvm.Tensor): Tensor of same type as x. dy is the adjoint of the derivative of x(y'),
                            namely, y' will be multiplied by dy.
        axis(int): scalar. Specifies which axis to reduce when compute max value of x.

    Returns:
        dx: The gradient. A akg.tvm.Tensor of same type as x.
    """

    check_list = ["float16", "float32"]
    dtype = x.dtype
    if not dtype in check_list:
        raise TypeError("softmax_grad_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    shape_x = [i.value for i in x.shape]
    shape_dy = [i.value for i in dy.shape]
    utils.check_shape(shape_x)
    utils.check_shape(shape_dy)
    if shape_x != shape_dy:
        raise ValueError("input tensors have different shapes")
    shape = shape_x

    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        raise ValueError("axis should be less than dimension")

    # y = softmax(x)
    y = Softmax(x, axis)

    # y' = y * (1 -  y)
    y_neg = akg.lang.ascend.vmuls(y, akg.tvm.const(-1.0, dtype=dtype))
    y_neg_add1 = akg.lang.ascend.vadds(y_neg, akg.tvm.const(1.0, dtype=dtype))
    y_grad = akg.lang.ascend.vmul(y, y_neg_add1)

    # dx = dy * y'
    dx = akg.lang.ascend.vmul(dy, y_grad)

    return dx
