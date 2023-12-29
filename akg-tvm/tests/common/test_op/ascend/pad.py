# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function:pad"""

from akg.ops.math import cast
from akg.utils.format_transform import get_const
from akg.topi.nn import pad as tvm_pad
import akg.utils as utils

# only consider last two axis, for matmul only


def auto_pad(data, target="cce"):
    shape = [get_const(x) for x in data.shape]
    assert len(shape) >= 2
    pad_shape = [(x + 15) // 16 * 16 for x in shape]
    paddings = [[0, 0] for _ in range(len(shape))]
    paddings[-1] = [0, pad_shape[-1] - shape[-1]]
    paddings[-2] = [0, pad_shape[-2] - shape[-2]]
    return pad(data, paddings, 'constant')


def pad(data, paddings, padtype, target="cce"):
    """add paddings to the tensor
    :shape: The shape of the tensor, now only support two dimension Tensor
    :paddings: The shape of the paddings, shape [N,2], N is the dimension of the tensor,
     For each dimension D of input, paddings[D, 0] indicates how many values to add before
     the contents of tensor in that dimension, and paddings[D, 1] indicates how many values to
     add after the contents of tensor in that dimension.
    :dtype: The type of the input, float16, float32
    :padtype: One of "CONSTANT", "REFLECT", or "SYMMETRIC".
    """
    # check shape
    utils.check_shape(data.shape)
    # check types
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_TYPES)
    # check padding types
    ptype_checklist = ['constant']
    if not (padtype in ptype_checklist):
        raise RuntimeError("pad_cce only support %s while padtype is %s" % (",".join(ptype_checklist), padtype))

    dtype = data.dtype
    if dtype == 'int8' or dtype == 'uint8':
        data = cast(data, "float16", target=target)

    rank = len(data.shape)
    pad_before = []
    pad_after = []
    for i in range(rank):
        pad_before.append(paddings[i][0])
        pad_after.append(paddings[i][1])
    b = tvm_pad(data, pad_before, pad_after=pad_after, name='B')

    if dtype == 'int8' or dtype == 'uint8':
        b = cast(b, dtype, target=target)
    return b
