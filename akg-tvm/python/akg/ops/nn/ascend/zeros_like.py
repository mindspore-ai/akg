#!/usr/bin/env python3
# coding: utf-8
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

"""operator dsl function: zeros_like"""

import akg.tvm
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor)
def ZerosLike(input, target=utils.CCE):
    """
    Generate an array of zeros.

    Args:
        input(tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor with the same type and shape as input.
    
    Supported Platforms:
        'Ascend'
    """
    dtype = input.dtype
    shape = [x.value for x in input.shape]
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])
    utils.check_shape(shape)
    output = akg.tvm.compute(input.shape, lambda *i: akg.tvm.const(0, input.dtype), name="output")
    return output
