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

"""operator dsl function:less"""

import akg.topi
import akg.tvm
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def less(data1, data2, target=utils.CCE):
    """
    compute tensor with smaller value in data1 and data2 elementwisely.

    Args:
        data1 (tvm.tensor.Tensor): Tensor of type float16, float32 and int32.
        data2 (tvm.tensor.Tensor): Tensor of type float16, float32 and int32.

    Returns:
        tvm.tensor.Tensor. If data1 less than data2, return True, else return False.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    utils.check_shape(data1.shape)
    utils.check_shape(data2.shape)

    # check types
    if target == utils.CCE:
        utils.elemwise_dtype_check(data1.dtype, data2.dtype, [utils.DtypeForDavinci.ALL_FLOAT,
            utils.DtypeForDavinci.INT32])
        # check runtime mode, and change dtype
        if product_is_mini() and data1.dtype != "float16":
            data1 = akg.topi.cast(data1, "float16")
            data2 = akg.topi.cast(data2, "float16")
        if (not product_is_mini()) and data1.dtype == "int32":
            data1 = akg.topi.cast(data1, "float32")
            data2 = akg.topi.cast(data2, "float32")
    res = akg.topi.less(data1, data2)
    return res
