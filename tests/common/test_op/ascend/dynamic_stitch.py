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

"""operator dsl function: dynamic_stitch"""
import akg.tvm
import akg.utils as utils

def dynamic_stitch(indices, data, target=utils.CCE):
    """
    The values in the data tensor are interleaved into a single tensor.

    Args:
        indices (tvm.tensor.Tensor): Tensor of type int32.
        data (tvm.tensor.Tensor): Tensor of type float16, float32, int32.
    Note:
        data's shape must be indices.shape + data_fragment_shape,  data_fragment_shape can be empty.

    Returns:
           tvm.tensor.Tensor, has the same type as data.
    """

    indices_shape = [x.value for x in indices.shape]
    data_shape = [x.value for x in data.shape]

    # Check params' shape
    utils.check_shape(indices_shape)
    utils.check_shape(data_shape)

    # Check dtype
    utils.ops_dtype_check(indices.dtype, utils.DtypeForDavinci.INT32)
    utils.ops_dtype_check(data.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])

    assert indices_shape == data_shape[:len(indices_shape)]
    length = 1
    for x in indices_shape:
        length *= x
    frac_shape = data_shape[len(indices_shape):]

    def get_indexes_from_flat(flat_index, shape):
        indexes = []
        p = 1
        for x in shape:
            p *= x
        r = flat_index
        for s in shape:
            p = p // s
            q = r // p
            indexes.append(q)
            r = r % p
        return tuple(indexes)

    def pick(index, s, *frac_i):
        indexes = get_indexes_from_flat(index, indices_shape)
        if len(frac_i) > 0:
            return akg.tvm.expr.Select(s == indices[indexes], akg.tvm.const(1, data.dtype), \
                                   akg.tvm.const(0, data.dtype)) * data[indexes + frac_i]
        return akg.tvm.expr.Select(s == indices[indexes], akg.tvm.const(1, data.dtype), \
                               akg.tvm.const(0, data.dtype)) * data[indexes]

    tmp = akg.tvm.compute([length, length] + frac_shape, lambda *i: pick(i[0], i[1], *i[2:]), name="tmp")
    reduce_axis = akg.tvm.reduce_axis((0, length))
    output = akg.tvm.compute([length] + frac_shape, lambda *i: akg.tvm.sum(tmp[tuple((reduce_axis,) + i)], axis=reduce_axis))
    return output
