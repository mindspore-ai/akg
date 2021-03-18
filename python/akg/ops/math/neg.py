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

"""operator dsl function: neg"""

import akg.topi
import akg.tvm
import akg
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def neg(data):
    """
    Computes negative value of input tensor.

    Args:
        data(tvm.tensor.Tensor): Tensor of type float16, float32, int32.

    Returns:
        tvm.tensor.Tensor of same type and shape as input tensor data.
    """
    vc_util.check_shape(data.shape)

    data_type = data.dtype
    vc_util.ops_dtype_check(data_type, [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT32])

    pone = akg.tvm.const(-1.0, dtype=data_type)
    res = akg.lang.cce.vmuls(data, pone)
    if data_type == "int32":
        res = akg.topi.cast(res, "int32")

    return res
