# Copyright 2020 Huawei Technologies Co., Ltd
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
from akg import topi
import akg.lang.cce
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.ops.math.cast import cast


@vc_util.check_input_type(akg.tvm.tensor.Tensor, (list, tuple))
def broadcast_to(x, shape):
    """
    Broadcast an tensor to a compatible shape.

    Args:
        x (tvm.tensor.Tensor): Tensor of type float32, float16, int8, uint8, int32
        shape (list, tuple): The shape of output tensor.

    Returns:
        An tvm.tensor.Tensor with the same type as x.

    """
    # check shape
    vc_util.check_shape(x)
    vc_util.check_shape(shape)

    # check dtype
    dtype = x.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_TYPES)

    # vector_dup instruction don't support int8 and uint8
    # It can be simplified by some methods, such as , "auto cast"
    x_shape = get_shape(x)
    if len(x_shape) == 1 and x_shape[0] == 1 and dtype in ["int8", "uint8"]:
        x = cast(x, "float16")

    res = topi.broadcast_to(x, shape)
    if res.dtype != dtype:
        res = cast(res, dtype)
    return res
