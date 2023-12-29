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

"""operator dsl function: assign_sub"""

import akg.tvm
import akg.topi
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def assign_sub(data1, data2):
    """
    Computes data1 - data2 elementwise.

    Args:
        data1 (tvm.tensor.Tensor): Tensor of type float16, float32, int32, int8, uint8.
        data2 (tvm.tensor.Tensor): Tensor of same shape and type as data1.

    Returns:
        Subtracted result, with same shape and type as input tensors.
    
    Supported Platforms:
        'Ascend'
    """
    dtype = data1.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_TYPES)
    utils.elemwise_dtype_check(data1.dtype, data2.dtype)
    utils.elemwise_shape_check(data1.shape, data2.shape)

    need_cast_dtype = ["int8", "uint8"]
    cast_type = "float16"
    if dtype in need_cast_dtype:
        data1 = akg.topi.cast(data1, cast_type)
        data2 = akg.topi.cast(data2, cast_type)

    res = akg.topi.subtract(data1, data2)

    if dtype in need_cast_dtype:
        if dtype == "uint8":
            cons = akg.tvm.const(256, dtype=cast_type)
            res = akg.tvm.compute(res.shape,
                                  lambda *indice:
                                  akg.tvm.expr.Select(res(*indice) < 0,
                                                      res(*indice) + cons,
                                                      res(*indice)),
                                  name="positive_res")
        res = akg.topi.cast(res, dtype)

    return res
