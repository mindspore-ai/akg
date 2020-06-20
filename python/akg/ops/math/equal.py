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

"""operator dsl function: equal"""
import akg.tvm
import akg
import akg.lang.cce

from akg.ops.math.cast import cast
from akg.ops.math.sub import sub
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def equal(input1, input2):
    """
    check whether input1 equals to input2.

    Args:
        input1 (tvm.tensor.Tensor): input argument has type float16, float32 and int32.
        input2 (tvm.tensor.Tensor): input argument has type float16, float32 and int32.

    Returns:
        tvm.tensor.Tensor. If input1 equal to input2 return True, else return False.
    """
    # check shapes
    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    shapes = [shape1, shape2]
    for _, shp in enumerate(shapes):
        vc_util.check_shape(shp)

    vc_util.ops_dtype_check([input1.dtype, input2.dtype],
                            [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT32,
                             vc_util.DtypeForDavinci.INT8, vc_util.DtypeForDavinci.UINT8])

    dtype = input1.dtype
    orig_dtype = dtype
    if utils.product_is_mini() and dtype != "float16":
        dtype = "float16"
    if (not utils.product_is_mini()) and dtype not in ("float16", "float32"):
        # for int32, if cast to float16, there may be overflow
        dtype = "float32"

    if orig_dtype == "float32" and dtype == "float16":
        input_sub = sub(input1, input2)
        input_sub = cast(input_sub, dtype)
        zero = akg.tvm.const(0.0, dtype)
        res = akg.topi.equal(input_sub, zero)
    else:
        input1 = cast(input1, dtype)
        input2 = cast(input2, dtype)
        res = akg.topi.equal(input1, input2)
    return res
