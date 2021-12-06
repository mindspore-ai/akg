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

"""operator dsl funnction:segment_max"""
import akg.tvm
from akg.utils import custom_tiling as ct_util
import akg.utils as utils
from akg.ops.math import ReduceMax
from akg.ops.array.ascend import Concat, Split

segment_max_set_dim_map = {
    str(((128, 256), "float32")): ((128, 128), (128, 128)),
    str(((128, 256), "float16")): ((128, 128), (256, 256)),
    str(((128, 16, 16), "float16")): ((128, 128), (16, 16), (16, 16)),
    str(((128, 1024), "float16")): ((128, 128), (256, 256)),
    str(((128, 1024), "float32")): ((128, 128), (128, 128)),
    str(((128, 64, 32), "float16")): ((128, 128), (64, 64), (1, 1)),

}


def segment_max_set_dim_func(data, segment_ids, num_segments):
    key = []
    key.append(tuple(data.shape))
    key.append(data.dtype)
    hash_key = str((tuple(key)))

    return ct_util.set_dims_by_key(hash_key, segment_max_set_dim_map), hash_key

def gen_ids(segment_ids):

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


@ct_util.reg_set_dim_func(segment_max_set_dim_func)
def segment_max(data, segment_ids, num_segments):
    """
    Computes the max value along segment_ids of a akg.tvm.tensor

    Args:
        data: akg.tvm.Tensor of type "float16", "float32" 
        segment_ids: akg.tvm.Tensor of type int32, sorted

    Returns:
        akg.tvm.Tensor of same shape and type as data

    """

    d_dtype = data.dtype
    utils.ops_dtype_check(d_dtype, utils.DtypeForDavinci.ALL_FLOAT)
    d_shape = [x.value for x in data.shape]
    utils.check_shape(d_shape)

    s_shape = segment_ids.shape
    utils.check_shape(s_shape)

    new_segment_ids, idx = gen_ids(segment_ids)

    output_shape = (1, ) + tuple(d_shape[len(s_shape):])
    zero_data = akg.tvm.compute(output_shape, lambda*i: akg.tvm.const(0.0, d_dtype), name = "zero")

    data_list = Split(data, new_segment_ids)
    out_n = num_segments

    out = []
    j = 0
    for i in range(0, out_n):

        if i in idx:
            tmp = ReduceMax(data_list[j], 0, True, target=utils.CCE)
            out.append(tmp)
            j = j + 1
        else:
            out.append(zero_data)

    res = Concat(out, 0)

    return res
