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

"""slice"""

import akg.tvm
from akg.utils import custom_tiling as ct_util
import akg.utils as utils
from akg.utils.format_transform import get_shape

slice_set_dim_map = {
    str(((8, 4718, 4), (0, 0, 0), (8, 3136, 4), "float32")): ((1, 1), (3136, 1), (4, 1)),
}


def slice_set_dim_func(data, begin, size):
    """setdim function"""
    shape = get_shape(data)

    key = str((tuple(shape), begin, size, data.dtype))
    if key in slice_set_dim_map.keys():
        return ct_util.set_dims(slice_set_dim_map[key]), key
    else:
        return "", key


@utils.check_input_type(akg.tvm.tensor.Tensor, (tuple, list), (tuple, list))
def slice(data, begin, size):
    """
    Extracts a slice from a tensor.

    Args:
        data (tvm.tensor.Tensor): Input data of type float16, float32, int32.
        begin (Union[tuple, list]): Specifies the start index of a slice.
        size (Union[tuple, list]): Specifies the size of a slice.
    
    Returns:
        tvm.tensor.Tensor, has the same type as input tensor data.
    """

    shape = get_shape(data)
    utils.check_shape(shape)
    utils.check_equal("len(shape)", "len(begin)", len(shape), len(begin))
    utils.check_equal("len(shape)", "len(size)", len(shape), len(size))
    utils.ops_dtype_check(data.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])

    dim_info, _ = slice_set_dim_func(data, begin, size)
    attrs = {"dim": dim_info}

    out_shape = [size[i] if size[i] > 0 else shape[i] - begin[i] for i in range(len(shape))]

    def slice_index(*inputs):
        return [begin[i] + inputs[i] for i in range(len(inputs))]

    res = akg.tvm.compute(out_shape, lambda *i: data(*slice_index(*i)), name='res')

    return res, attrs
