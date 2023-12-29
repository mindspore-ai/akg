# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: another implementation of sum"""

import akg
import akg.utils as utils
from akg.utils import format_transform as ft_util
from akg.utils.format_transform import get_shape
from ..cast import cast
from ..sum import sum


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple, int, type(None)), (bool, type(None)), (str, type(None)))
def sum_v2(inputs, axis=None, keepdims=True, target=utils.CCE):
    """
    another implementation of sum with topi api.

    Supported Platforms:
        'Ascend'
    """
    if target != utils.CCE:
        raise RuntimeError('operator not supported on %s' % utils.get_backend(target))

    dtype = inputs.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    axis = ft_util.refine_reduce_axis(inputs, axis)
    utils.check_shape(inputs.shape)
    if not axis:
        output = akg.topi.identity(inputs)
    else:
        if dtype == "float16":
            step_sum = cast(inputs, "float32", target)
        else:
            step_sum = inputs

        step_sum = akg.topi.sum(step_sum, axis=axis, keepdims=keepdims)

        if dtype == "float16":
            output = cast(step_sum, "float16", target)
        else:
            output = step_sum
    return output


def sum_by_shape(broadcast_data, original_shape, target=utils.CCE):
    """
    sum the broadcast_data by original shape; gradient for Broadcast.

    Supported Platforms:
        'Ascend'
    """
    if target != utils.CCE:
        raise RuntimeError('operator not supported on %s' % utils.get_backend(target))

    broadcast_shape = get_shape(broadcast_data)
    original_shape = get_shape(original_shape)
    if broadcast_shape == original_shape:
        return broadcast_data
    if original_shape == [1]:
        data = sum(broadcast_data, target=target)
        return data

    utils.broadcast_check(original_shape, broadcast_shape)
    axis_len = len(broadcast_shape) - len(original_shape)
    if axis_len > 0:
        axis = list(range(axis_len))
        broadcast_data = sum(broadcast_data, axis, False, target=target)
        broadcast_shape = get_shape(broadcast_data)

    axis = []
    for i, _ in enumerate(original_shape):
        if original_shape[i] != broadcast_shape[i]:
            axis.append(i)
    res = sum(broadcast_data, axis, True, target=target)[0] if axis else broadcast_data
    return res
