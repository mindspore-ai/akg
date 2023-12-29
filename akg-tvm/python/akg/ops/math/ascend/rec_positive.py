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

"""operator dsl function: rec_positive"""
import akg
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def RecPositive(x, target=utils.CCE):
    """
    Calculate 1/x when data in x are all positive, used by dsl tanh and focalloss_grad.

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32. data in x must be positive.

    Returns:
        tvm.tensor.Tensor, the same type as inputs.
    
    Supported Platforms:
        'Ascend'
    """

    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    need_conv = product_is_mini() and x.dtype == "float32"
    x_fp16 = x
    if need_conv:
        x_fp16 = x.astype("float16")
    log = akg.topi.log(x_fp16)
    neg_log = akg.topi.negative(log)
    res = akg.lang.ascend.vexp(neg_log)
    return res.astype(x.dtype) if need_conv else res
