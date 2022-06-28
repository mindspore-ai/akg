# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

import akg.utils as utils
from .abs import abs
from .sum import sum


def abs_sum(inputs, axis=None, keepdims=False, target=utils.CCE):
    """
    Computes the sum of absolute value of inputs tensor along axis.

    Args:
        inputs: The inputs akg.tvm.tensor.
        axis: An integer, specifies the dimensions to reduce when performing the sum operation.
        keepdims: A boolean, if True, retains reduced dimensions with length 1, default value is False.

    Returns:
        A akg.tvm.Tensor of same type as inputs.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    input_abs = abs(inputs, target)
    return sum(input_abs, axis, keepdims, target=target)
