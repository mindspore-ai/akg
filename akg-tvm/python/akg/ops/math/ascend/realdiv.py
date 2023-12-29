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

"""operator dsl function:realdiv"""

import akg.topi
import akg.utils as utils
from akg.utils.dsl_create import produce_shapes


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def RealDiv(input1, input2, target=utils.CCE):
    """
    Returns input1 / input2 element-wise for real types.

    Note:
        Realdiv supports broadcasting.

    Args:
        input1 (tvm.tensor.Tensor): Tensor of type float16, float32.
        input2 (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has the same type of input1 and shaped by broadcasting.
    
    Supported Platforms:
        'Ascend'
    """
    utils.ops_dtype_check([input1.dtype, input2.dtype], utils.DtypeForDavinci.ALL_FLOAT)
    utils.elemwise_dtype_check(input1.dtype, input2.dtype)

    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)

    utils.auto_broadcast_check(shape1, shape2)
    n_shape1, n_shape2, out_shape = produce_shapes(shape1, shape2)

    if n_shape1 != out_shape:
        input1_cast = akg.topi.broadcast_to(input1, out_shape)
    else:
        input1_cast = input1
    if n_shape2 != out_shape:
        input2_cast = akg.topi.broadcast_to(input2, out_shape)
    else:
        input2_cast = input2

    res = akg.topi.divide(input1_cast, input2_cast)
    return res
