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

"""operator dsl function: reshape"""

import akg
import akg.topi
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from functools import reduce


@vc_util.check_input_type(akg.tvm.tensor.Tensor, (list, tuple))
def reshape(data, out_shape):
    """
    Rearranges input tensor data to new shape out_shape.

    Args:
        data (tvm.tensor.Tensor): The tensor to be reshaped.
        out_shape (list, tuple): The new shape applied on the input tensor data,
                                should be compatible with the original shape of data.

    Returns:
        The reshaped akg.tvm.tensor of same type as input tensor data.
    """

    data_shape = data.shape
    vc_util.check_shape(data_shape)

    in_shape = get_shape(data)
    out_shape = list(out_shape)

    if -1 in out_shape:
        access_size = 1
        for i, o_shape in enumerate(out_shape):
            if -1 != o_shape:
                access_size *= o_shape
            else:
                hit_idx = i
        ori_size = reduce(lambda x, y: x * y, in_shape)
        if ori_size % access_size != 0:
            raise ValueError(("Invalid out_shape ({})".format(out_shape)))

        out_shape[hit_idx] = int(ori_size / access_size)

    res = akg.topi.reshape(data, out_shape)
    return res
