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

"""operator dsl function:logsoftmax"""
import akg
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape, refine_reduce_axis


def logsoftmax_op(data, shape, axis):
    max_data = akg.lang.cce.reduce_max(data, axis=axis, keepdims=True)
    max_broadcast = akg.lang.cce.broadcast(max_data, shape)
    data_sub = akg.lang.cce.vsub(data, max_broadcast)
    data_exp = akg.lang.cce.vexp(data_sub)
    data_expsum = akg.lang.cce.sum(data_exp, axis, keepdims=True)
    logexpsum = akg.lang.cce.vlog(data_expsum)
    data_logexpsum_broadcast = akg.lang.cce.broadcast(logexpsum, shape)
    out = akg.lang.cce.vsub(data_sub, data_logexpsum_broadcast)
    return out


def logsoftmax(inputs, axis):
    """
    Activation function, computes log softmax.

    Args:
        inputs: Tensor.
        axis: On which dimension log softmax is performed.

    Return:
        Tensor, which has the same shape and type as input.
    """
    dtype = inputs.dtype
    vc_util.check_shape(inputs.shape)
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    axis = refine_reduce_axis(inputs, axis)
    if isinstance(axis, (list, tuple)):
        if len(axis) != 1:
            raise RuntimeError("Reduce axis for logsoftmax op must br 1-dimension, while current is %d-dimension"
                               % (len(axis)))
        axis = axis[0]
    out = logsoftmax_op(inputs, inputs.shape, axis)
    attr_map = {"pragma_modshift": 1, "disable_cse": 1}
    return out, attr_map
