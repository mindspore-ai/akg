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

"""operator dsl function:floor"""

import akg
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def floor(data, target=utils.CCE):
    """
    Returns element-wise largest integer not greater than x.

    Args:
        data (akg.tvm.tensor.Tensor): Tensor of type float16, and float32

    Returns:
        akg.tvm.tensor.Tensor, has the same shape as data and type of int32.

    Supported Platforms:
        'Ascend'
    """
    if target != utils.CCE:
        raise RuntimeError('operator not supported on %s' % utils.get_backend(target))
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    shape = [x.value for x in data.shape]
    utils.check_shape(shape)

    if product_is_mini() and data.dtype == "float32":
        # solve the problem of 87==floor(86.9996) when high_precision is needed.
        # problem is caused by such as fp16(86.9996)==87.
        # detect problem by fp32(86.9996) - fp32(floor(fp16(86.9996))) < 0

        # floor could only apply on float16
        data_fp16 = akg.lang.ascend.cast_to(data, "float16")
        floor_data = akg.lang.ascend.floor(data_fp16)
        floor_fp16 = akg.lang.ascend.cast_to(floor_data, "float16")
        floor_fp32 = akg.lang.ascend.cast(floor_fp16, "float32")

        # if diff=1e-7, we cannot get right sign of fp16(diff)
        # but we can get right sign of 10000*diff = 1e-3, which has the same
        # sign as diff
        diff = (data - floor_fp32) * 10000
        diff_fp16 = akg.lang.ascend.cast_to(diff, "float16")

        # if diff < 0 and floor == ceil, then it's 87 = floor(86.99999)
        res = akg.tvm.compute(shape,
                              lambda *i: akg.tvm.expr.Select(
                                  diff_fp16(*i) < akg.tvm.const(0, "float16"),
                                  floor_fp16(*i) - akg.tvm.const(1, "float16"),
                                  floor_fp16(*i)),
                              name="res")

        res = akg.lang.ascend.cast_to(res, "int32")
    else:
        res = akg.lang.ascend.floor(data)

    return res
