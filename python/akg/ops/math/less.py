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

"""operator dsl function:less"""

import akg.topi
import akg.tvm
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def less(data1, data2):
    """
    compute tensor with smaller value in data1 and data2 elementwisely.

    Args:
        data1 (tvm.tensor.Tensor): Tensor of type float16, float32 and int32.
        data2 (tvm.tensor.Tensor): Tensor of type float16, float32 and int32.

    Returns:
        tvm.tensor.Tensor. If data1 less than data2, return True, else return False.
    """

    vc_util.check_shape(data1.shape)
    vc_util.check_shape(data2.shape)

    # check types
    vc_util.elemwise_dtype_check(data1.dtype, data2.dtype, [vc_util.DtypeForDavinci.ALL_FLOAT,
                                                            vc_util.DtypeForDavinci.INT32])

    # check runtime mode, and change dtype
    if utils.product_is_mini() and data1.dtype != "float16":
        data1 = akg.topi.cast(data1, "float16")
        data2 = akg.topi.cast(data2, "float16")
    if (not utils.product_is_mini()) and data1.dtype == "int32":
        data1 = akg.topi.cast(data1, "float32")
        data2 = akg.topi.cast(data2, "float32")

    res = akg.topi.less(data1, data2)
    return res
