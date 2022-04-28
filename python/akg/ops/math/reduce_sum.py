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

"""operator dsl function: sum"""

import akg.topi
import akg.tvm
from akg.utils.format_transform import refine_reduce_axis
from akg.lang.ascend import sum as ascend_sum
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple, int, type(None)), (bool, type(None)), (str, type(None)))
def reduce_sum(inputs, axis=None, keepdims=False, target=utils.CCE):
    """
    Compute the sum of elements across dimensions of a tensor.

    Args:
        inputs (tvm.tensor.Tensor): Tensor.
        axis (Union[list, tuple, int, None]): If the list or tuple is empty, the axis equal to None.
        keepdims (bool): If keepdims equal to True, the result shape length is same to input shape length.

    Returns:
        tvm.tensor.Tensor, has same type as input. If keepdims is True, all reduced dimensions are retained
        with length 1, else these reduced axis will be eliminate.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return ascend_sum(inputs, axis, keepdims)
    axis = refine_reduce_axis(inputs, axis)
    utils.check_shape(inputs.shape)

    in_dtype = inputs.dtype
    if in_dtype == 'float16':
        inputs = akg.topi.cast(inputs, 'float32')

    output = akg.topi.sum(inputs, axis=axis, keepdims=keepdims)

    if in_dtype == 'float16':
        output = akg.topi.cast(output, 'float16')

    return output
