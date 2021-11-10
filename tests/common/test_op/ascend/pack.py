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

"""operator dsl function: pack"""
import akg
from akg import topi
from akg.utils.format_transform import get_shape
import akg.utils as utils


@utils.check_input_type(((tuple, list), akg.tvm.tensor.Tensor), int)
def pack(x, axis):
    """
    Concatenates tensors along one dimension.

    Args:
        x (Union[tuple, list]): Inprut tensor. Support int8, uint8,
                                int16, uint16, int32, uint32, int64, uint64,
                                float16, float32.
        axis (int): in the range [-rank(x), rank(x))

    Returns:
        tvm.tensor.Tensor
    """
    for _, tensor in enumerate(x):
        shape_tensor = get_shape(tensor)
        utils.check_shape(shape_tensor)
        utils.ops_dtype_check(
            tensor.dtype, [
                utils.DtypeForDavinci.BOOL,
                utils.DtypeForDavinci.INT8, utils.DtypeForDavinci.INT16,
                utils.DtypeForDavinci.INT32, utils.DtypeForDavinci.INT64,
                utils.DtypeForDavinci.UINT8, utils.DtypeForDavinci.UINT16,
                utils.DtypeForDavinci.UINT32, utils.DtypeForDavinci.UINT64,
                utils.DtypeForDavinci.FLOAT16, utils.DtypeForDavinci.FLOAT32
            ])

    if (axis < -len(get_shape(x[0])) - 1) or (axis > len(get_shape(x[0]))):
        raise RuntimeError(
            "pack axis must be in [-%d , %d), "
            "actual is %d" % (
                len(get_shape(x[0])) + 1, len(get_shape(x[0])) + 1, axis))

    if axis == -1 or axis == len(get_shape(x[0])):
        raise RuntimeError("pack does not support the last dimension")
    if axis < -1:
        axis = axis + 1

    return topi.concatenate(tuple(x), axis=axis)
