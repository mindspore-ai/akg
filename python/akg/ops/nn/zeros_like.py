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

"""operator dsl function: zeros_like"""

import akg.tvm
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def zeros_like(input):
    """
    Generate an array of zeros.

    Args:
        input(tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor with the same type and shape as input.
    """
    dtype = input.dtype
    shape = [x.value for x in input.shape]
    vc_util.ops_dtype_check(dtype, [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT32])
    vc_util.check_shape(shape)
    output = akg.tvm.compute(input.shape, lambda *i: akg.tvm.const(0, input.dtype), name="output")
    return output
