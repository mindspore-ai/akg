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

"""operator dsl function: logical_or"""
import akg.tvm
import akg.topi
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def logical_or(input1, input2, target=utils.CCE):
    """
    Compute logical_or of input1 and input2.

    Args:
        input1 (tvm.tensor.Tensor): Tensor.
        input2 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor. logical_or of input1 and input2.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    utils.elemwise_dtype_check(input1.dtype, input2.dtype)

    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)

    res = akg.topi.logical_or(input1, input2)
    return res
