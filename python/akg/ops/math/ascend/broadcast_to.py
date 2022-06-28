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

"""operator dsl function: broadcast_to"""
import akg
import akg.utils as utils
from akg import topi
from akg.utils.format_transform import get_shape
from ..cast import cast


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), (str, type(None)))
def broadcast_to(x, shape, target=utils.CCE):
    """
    Broadcast an tensor to a compatible shape.

    Args:
        x (tvm.tensor.Tensor): Tensor of type float32, float16, int8, uint8, int32
        shape (list, tuple): The shape of output tensor.

    Returns:
        An tvm.tensor.Tensor with the same type as x.

    Supported Platforms:
        'Ascend'
    """
    # check shape
    utils.check_shape(x)
    utils.check_shape(shape)

    # check dtype
    dtype = x.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_TYPES)

    # vector_dup instruction don't support int8 and uint8
    # It can be simplified by some methods, such as , "auto cast"
    x_shape = get_shape(x)
    if len(x_shape) == 1 and x_shape[0] == 1 and dtype in ["int8", "uint8"]:
        x = cast(x, "float16", target)

    res = topi.broadcast_to(x, shape)
    if res.dtype != dtype:
        res = cast(res, dtype, target)
    return res
