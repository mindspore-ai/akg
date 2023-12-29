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

"""space to batch"""

import akg.tvm
import akg.topi.nn as nn
from functools import reduce
from akg.utils import custom_tiling as ct_util
import akg.utils as utils

space_to_batch_nd_set_dim_map = {
    str(((1, 33, 33, 1024), "float16", (2, 2), ((5, 0), (5, 0)))): ((16, 1), (4, 1), (34, 1), (38, 1)),
    str(((1, 33, 33, 1536), "float16", (2, 2), ((5, 0), (5, 0)))): ((16, 1), (4, 1), (34, 1), (38, 1)),
    str(((1, 3, 3, 960), "float16", (2, 2), ((5, 0), (5, 0)))): ((480, 1), (4, 1), (7, 1), (8, 1)),
    str(((2, 16, 22, 3), "float32", (4, 3), ((2, 2), (1, 1)))): ((3, 1), (25, 1), (16, 1), (24, 1)),
}


def space_to_batch_nd_set_dim_func(data, block, paddings):
    """Setdim function"""
    key = []
    key.append(tuple(data.shape))
    key.append(data.dtype)
    key.append(tuple(block))
    key.append(tuple(paddings))
    hash_key = str(tuple(key))

    if hash_key in space_to_batch_nd_set_dim_map.keys():
        return ct_util.set_dims(space_to_batch_nd_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key


def _get_pad_before_and_after(n, m, paddings, pad_before=None, pad_after=None):
    """Get the padding shape"""
    for i in range(n):
        if i >= 1 and i < 1 + m:
            pad_before.append(paddings[i - 1][0])
            pad_after.append(paddings[i - 1][1])
        else:
            pad_before.append(0)
            pad_after.append(0)

def check_inputs(data, block_shape, paddings):
    """check input shape and types"""
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_TYPES)
    utils.check_shape(data, tensor_name="data")
    utils.check_shape(block_shape, tensor_name="block_shape")
    if not isinstance(paddings, (list, tuple)):
        raise RuntimeError("paddings must be a 2D list or tuple.")
    for cs in paddings:
        if not isinstance(cs, (list, tuple)) or len(cs) != 2:
            raise RuntimeError("paddings must be a 2D list or tuple and the 2nd dim has length 2.")
        if cs[0] < 0 or cs[1] < 0:
            raise RuntimeError("all values in paddings must be greater than or equal to zero.")
    utils.check_equal("length of block_shape", "length of paddings", len(block_shape), len(paddings))

@utils.check_input_type(akg.tvm.tensor.Tensor, (tuple, list), (tuple, list))
def space_to_batch_nd(data, block_shape, paddings):
    """
    The N-D version of SpaceToBatch.
    
    Zero padding, then rearranging spatial data blocks into batch.
    
    Args:
        data (tvn.tensor.Tensor): Spacial data of type float16, float32, int8, uint8, int32.
        block_shape (Union[tuple, list]): 1-D shape of length `L`.
        paddings (Union[tuple, list]): 2-D list of shape `[L][2]`, all values must be greater than or equal to 0.
    
    Returns:
        tvn.tensor.Tensor, has the same type as inputs
    """

    check_inputs(data, block_shape, paddings)
    dim_info, _ = space_to_batch_nd_set_dim_func(data, block_shape, paddings)
    attrs = {"dim": dim_info}

    block_shape = list(block_shape)

    pad_before = []
    pad_after = []
    n = len(data.shape)
    m = len(block_shape)
    _get_pad_before_and_after(n, m, paddings, pad_before, pad_after)
    prod_of_block_shape = reduce(lambda x, y: x * y, block_shape)

    data_shape_padded = nn.pad(data, pad_before, pad_after)

    M = len(block_shape)
    batch = data_shape_padded.shape[0]
    spatial_shape = data_shape_padded.shape[1:1 + M]
    remain_shape = data_shape_padded.shape[1 + M:]

    oshape = [batch * prod_of_block_shape] + \
        [dim // bsize for dim, bsize in zip(spatial_shape, block_shape)] + remain_shape

    def map_index(*index):
        ibatch = index[0] % batch
        ispatial = list(index[1:1 + M])
        iremain = list(index[1 + M:])

        coef = index[0] // batch
        for i in reversed(range(M)):
            ispatial[i] = coef % block_shape[i] + index[1 + i] * block_shape[i]
            coef = coef // block_shape[i]

        return [ibatch] + ispatial + iremain

    output = akg.tvm.compute(oshape, lambda *i: data_shape_padded(*map_index(*i)), name='output')
    return output, attrs
