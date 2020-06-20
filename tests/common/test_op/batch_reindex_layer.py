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

"""operator dsl function:batch_reindex_layer"""

import akg.tvm
from akg.utils import validation_check as vc_util


def batch_reindex_layer(input_data, permut):
    """
    Input data is reindexed along shape[0].

    Shape[0] is regarded as "index" and to be "reindexed".

    Args:
        input_data: Tensor.
        permut: Array list. Used to select input tensor from shape[1:].

    Returns:
        Tensor, has the same type as input_data, and shape[0] is changed by permut.
    """

    # check dtype
    dtype = input_data.dtype
    check_list = ["float16", "float32", "int32", "int8", "uint8"]
    if not dtype.lower() in check_list:
        raise RuntimeError("batch_reindex_layer only support %s while dtype is %s"
                           % (",".join(check_list), dtype))

    # check shape
    shape = input_data.shape
    vc_util.check_shape(shape)

    # check permut
    for index in permut:
        if index < 0 or index >= int(shape[0]):
            raise RuntimeError("index in list should be greater than 0 and less than shape[0].")

    oshape = [len(permut),] + input_data.shape[1:]

    def map_index(*index):
        idx = index[0]
        in_n = permut[0]
        for i in range(len(permut))[1:]:
            in_n = akg.tvm.expr.Select((idx == i), permut[i], in_n)
        return [in_n,] + list(index[1:])

    output_data = akg.tvm.compute(oshape, lambda *i: input_data(*map_index(*i)), name='output')

    return output_data
