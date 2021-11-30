# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function:logsoftmax_grad"""
import akg
from akg.utils import kernel_exec as utils
from akg.utils import custom_tiling as ct_util
from akg.utils import validation_check as vc_util
import akg.topi

logsoftmaxgrad_set_dim_map = {
    str(((160, 30522), 1, "float32")): ((1, 1), (10174, 1)),
    str(((1280, 30522), 1, "float32")): ((1, 1), (10174, 1)),
    str(((1280, 21128), 1, "float32")): ((1, 1), (2641, 1)),
}




def logsoftmaxgrad_set_dim_func(Y, dY, axis):
    shape = [x.value for x in Y.shape]

    if axis < 0:
        axis += len(shape)

    key = str((tuple(shape), axis, dY.dtype))

    if key in logsoftmaxgrad_set_dim_map.keys():
        return ct_util.set_dims(logsoftmaxgrad_set_dim_map[key]), key
    else:
        return "", key


@ct_util.reg_set_dim_func(logsoftmaxgrad_set_dim_func)
def logsoftmax_grad(Y, dY, axis):
    """
    Computes the back propagation gradients by chain rule.

    Args:
        Y: Tensor, holds the logsoftmax activation output.
        dY: Tensor, holds the initial gradients.
        axis: Integer, on which dimension the softmax is applied.

    Returns:
        Tensor, the overall gradients.
    """
    shape = [x.value for x in Y.shape]
    vc_util.check_shape(shape)
    dtype = Y.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    if axis == -1:
        axis = len(shape) + axis
    if axis >= len(shape):
        raise RuntimeError("axis should be less than dimension")
    if axis < -1:
        raise RuntimeError("negative axis only support -1, please specify the axis in positive value")

    softmax = akg.topi.exp(Y)
    dy_sum = akg.lang.cce.sum(dY, axis=axis)
    dy_sum_broadcast = akg.lang.cce.broadcast(dy_sum, shape)
    mul_result = akg.lang.cce.vmul(softmax, dy_sum_broadcast)
    res = akg.lang.cce.vsub(dY, mul_result)
    attrs = {"pragma_modshift": 1}
    return res, attrs
