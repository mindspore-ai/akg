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

"""operator dsl function: concat"""

import akg
import akg.utils as utils

min_size = 1

@utils.check_input_type((list, tuple), int, (str, type(None)))
def Concat(data, axis, target=utils.CCE):
    """
    Concatenates data along the dimension set by axis.

    Args:
        data (Union[list, tuple]): list or tuple of tvm.tensor.Tensor of type float16, float32, int32, int8, uint8
        axis (int): Specifies the axis along which to concatenate. Must be in the range [-rank(data), rank(data))

    Returns:
        tvm.tensor.Tensor of same type as data.
    """
  
    data_size = len(data)
    if data_size < min_size:
       raise RuntimeError("The size of data must be greater equal 1")

    dtype = data[0].dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_TYPES)

    shape_0 = data[0].shape
    utils.check_shape(shape_0)
    if axis < 0:
        axis += len(shape_0)

    for i in range(1, data_size):
        shape_i = data[i].shape
        utils.check_shape(shape_i)
        if len(shape_i) != len(shape_0):
            raise ValueError("Input tensors must have same dimensions.")

    res = akg.lang.ascend.concat(data, axis)
    return res
