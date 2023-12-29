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

"""operator dsl function: reshape"""

from functools import reduce
import akg
import akg.topi
import akg.utils as  utils
from akg.utils.format_transform import get_shape
from akg.utils import dynamic_shape as ds


def get_out_shape(in_shape, out_shape):
    """Computes output shape."""
    access_size = 1
    for i, o_shape in enumerate(out_shape):
        if -1 != o_shape:
            access_size *= o_shape
        else:
            hit_idx = i
    ori_size = reduce(lambda x, y: x * y, in_shape)
    if ori_size % access_size != 0:
        raise ValueError(("Invalid out_shape ({})".format(out_shape)))

    out_shape[hit_idx] = int(ori_size // access_size)
    return out_shape


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), (str, type(None)))
def reshape(data, out_shape, target=utils.CUDA):
    """
    Rearranges input tensor data to new shape out_shape.

    Args:
        data (tvm.tensor.Tensor): The tensor to be reshaped.
        out_shape (list, tuple): The new shape applied on the input tensor data,
                                should be compatible with the original shape of data.

    Returns:
        The reshaped akg.tvm.tensor of same type as input tensor data.

    Supported Platforms:
        'Ascend', 'GPU'
    """
    if target == utils.CCE:
        return _reshape_ascend(data, out_shape)
    data_shape = data.shape
    utils.check_shape(data_shape)

    in_shape = get_shape(data)
    out_shape = list(out_shape)

    if -1 in out_shape:
        out_shape = get_out_shape(in_shape, out_shape)

    res = akg.topi.reshape(data, out_shape)
    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), (str, type(None)))
def _reshape_ascend(data, out_shape):
    """
    Rearranges input tensor data to new shape out_shape.

    Args:
        data (tvm.tensor.Tensor): The tensor to be reshaped.
        out_shape (list, tuple): The new shape applied on the input tensor data,
                                should be compatible with the original shape of data.

    Returns:
        The reshaped akg.tvm.tensor of same type as input tensor data.

    Supported Platforms:
        'Ascend'
    """
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.INT32.value + utils.DtypeForDavinci.ALL_FLOAT.value)

    data_shape = data.shape
    utils.check_shape(data_shape)

    in_shape = get_shape(data)
    out_shape = list(out_shape)
    is_dynamic = ds.shape_is_dynamic(data)

    if -1 in out_shape:
        out_shape = get_out_shape(in_shape, out_shape)
    else:
        if not is_dynamic:
            if reduce(lambda x, y: x * y, in_shape) != reduce(lambda x, y: x * y, out_shape):
                raise ValueError("the total length of out_shape is not equal to the in_shape")

    inputs = akg.tvm.compute(in_shape, lambda *indice: data(*indice), name="inputs")
    res = akg.topi.reshape(inputs, out_shape)
    output = akg.tvm.compute(out_shape, lambda *indice: res(*indice), name="reshape")
    return output
