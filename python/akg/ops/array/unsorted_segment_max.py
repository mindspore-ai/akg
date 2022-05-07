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

"""operator dsl function: unsorted_segment_max"""
import akg.tvm
from akg.ops.math import reduce_max
from akg.ops.array.ascend import Split, Concat
import akg.utils as utils
from akg.utils import custom_tiling as ct_util


unsorted_segment_max_set_dim_map = {
    '((128, 32), "float16")': ((128, 128), (32, 32)),
    '((512, 256), "float16")': ((512, 512), (8, 8)),
    '((128, 128, 16, 16), "float16")': ((128, 128), (16, 16), (16, 16)),
    '((1024, 1024), "float16")': ((1024, 1024), (32, 32)),
    '((256, 1024), "float32")': ((256, 256), (4, 4)),
    '((128, 64, 32), "float16")': ((128, 128), (64, 64), (1, 1)),

}


def unsorted_segment_max_set_dim_func(data, segment_ids, num_segments):
    """Sets dim for UnsortedSegmentMax."""
    key = []
    key.append(tuple(data.shape))
    key.append(data.dtype)
    hash_key = str((tuple(key)))

    return ct_util.set_dims_by_key(hash_key, unsorted_segment_max_set_dim_map), hash_key


def gen_ids(segment_ids):
    """Generates ids."""
    segment_ids = list(segment_ids)

    res = []
    index = []

    begin = 0
    value = segment_ids[0]

    for i in range(1, len(segment_ids)):
        if segment_ids[i] != value:

            res.append(i - begin)
            index.append(value)

            begin = i
            value = segment_ids[i]

    res.append(len(segment_ids) - begin)
    index.append(segment_ids[-1])
    return res, index


def get_data_from_data_list(idx, data_list, i):
    """Gets data from data list."""
    data = []
    for (tmp, tmp_data) in zip(idx, data_list):
        if tmp == i:
            data.append(tmp_data)
    return data


def split_new(data, new_segment_ids, idx, num_segments):
    """Splits new."""
    data_list = Split(data, new_segment_ids)
    if not isinstance(data_list, (list, tuple)):
        data_list = [data_list]
    new_idx = []
    out = []
    for i in range(0, num_segments):

        if i in idx:
            new_idx.append(i)
            temp_data = get_data_from_data_list(idx, data_list, i)
            out.append(Concat(temp_data, 0))

    return out, new_idx


@ct_util.reg_set_dim_func(unsorted_segment_max_set_dim_func)
def unsorted_segment_max(data, segment_ids, num_segments, target=utils.CCE):
    """
    Computes the max value along segment_ids of a akg.tvm.Tensor

    Args:
        data: akg.tvm.Tensor of type float16, float32
        segment_ids: akg.tvm.Tensor of type int32, shape is a prefix of input_data.shape.
        num_segments: the number of classes in segment_ids

    Returns:
        akg.tvm.Tensor of same type as input_data

    Supported Platforms:
        'Ascend'
    """
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)

    d_shape = [x.value for x in data.shape]
    utils.check_shape(d_shape)

    utils.check_shape(segment_ids.shape)

    new_segment_ids, idx = gen_ids(segment_ids)

    output_shape = (1, ) + tuple(d_shape[len(segment_ids.shape):])

    zero_data = akg.tvm.compute(output_shape, lambda*i: akg.tvm.const(0.0, data.dtype), name="zero")

    data_list, new_idx = split_new(data, new_segment_ids, idx, num_segments)

    out = []
    j = 0
    for i in range(0, num_segments):
        if i in new_idx:
            tmp = reduce_max(data_list[j], 0, True, target)
            out.append(tmp)
            j = j + 1
        else:
            out.append(zero_data)

    return Concat(out, 0)
