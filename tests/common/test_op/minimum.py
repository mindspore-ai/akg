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

"""operator dsl function:minimum"""
import akg.topi
from akg.ops.math.cast import cast
from akg.utils import validation_check as vc_util


def minimum(input1, input2):
    """
    Return the min value of two tensors element-wise.

    Note:
        minimum supports broadcasting.

    Args:
        input1: Tensor.
        input2: Tensor. Has the same type as input1.

    Returns:
        Tensor, has the same type as inputs.
    """

    vc_util.ops_dtype_check([input1.dtype, input2.dtype], vc_util.DtypeForDavinci.ALL_TYPES)
    vc_util.elemwise_dtype_check(input1.dtype, input2.dtype)
    dtype = input1.dtype

    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    vc_util.check_shape(shape1)
    vc_util.check_shape(shape2)

    vc_util.auto_broadcast_check(shape1, shape2)

    if dtype in ("int8", "uint8"):
        input1 = cast(input1, "float16")
        input2 = cast(input2, "float16")
    res = akg.topi.minimum(input1, input2)
    if dtype in ("int8", "uint8"):
        res = cast(res, dtype)

    return res
