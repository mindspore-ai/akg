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

"""operator dsl function: minimum"""
import akg.tvm
import akg.topi
import akg.utils as utils
from .cast import cast


def minimum(input1, input2, target=utils.CCE):
    """
    Return the min value of two tensors element-wise.

    Note:
        minimum supports broadcasting.

    Args:
        input1: Tensor.
        input2: Tensor. Has the same type as input1.

    Returns:
        Tensor, has the same type as inputs.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    utils.ops_dtype_check([input1.dtype, input2.dtype], utils.DtypeForDavinci.ALL_TYPES)
    utils.elemwise_dtype_check(input1.dtype, input2.dtype)
    dtype = input1.dtype

    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)

    utils.auto_broadcast_check(shape1, shape2)

    need_cast = True if target == utils.CCE and dtype in ["int8", "uint8"] else False
    if need_cast:
        input1 = cast(input1, "float16", target)
        input2 = cast(input2, "float16", target)
    res = akg.topi.minimum(input1, input2)
    if need_cast:
        res = cast(res, dtype, target)
    return res
