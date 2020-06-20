#!/usr/bin/env python3
# coding: utf-8
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

"""operator dsl function: abs"""

import akg.tvm
from akg.utils.validation_check import ops_dtype_check, check_shape, DtypeForDavinci, check_input_type


@check_input_type(akg.tvm.tensor.Tensor)
def abs_value(in_data):
    """
    Compute absolute value of a tensor.

    Args:
        in_data (tvm.tensor.Tensor): Tensor of type float16, float32, int8, uint8, int32.

    Returns:
        tvm.tensor.Tensor of same type and shape as in_data.
    """
    # check type
    dtype = in_data.dtype
    ops_dtype_check(dtype, DtypeForDavinci.ALL_TYPES)

    # check shape
    check_shape(in_data.shape)

    need_cast_dtype = ["int8", "int32", "uint8"]

    if dtype in need_cast_dtype:
        in_data = akg.tvm.compute(in_data.shape, lambda *indice: in_data(*indice).astype("float16"), name='type_cast')

    output = akg.tvm.compute(in_data.shape, lambda *index: akg.tvm.abs(in_data(*index)), name='abs_value')

    if dtype in need_cast_dtype:
        output = akg.tvm.compute(in_data.shape, lambda *indice: output(*indice).astype(dtype), name='res')

    return output
