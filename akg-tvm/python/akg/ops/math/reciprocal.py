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

"""operator dsl function: reciprocal"""
import akg.tvm
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini


@utils.check_input_type(akg.tvm.tensor.Tensor, (bool, type(None)), (str, type(None)))
def reciprocal(data, high_precision=True, target=utils.CCE):
    """
    Computes the reciprocal of data element-wise.

    Args:
        data (list[tvm.tensor.Tensor]): a list of tvm.tensor.Tensor of type float16, float32.
        high_precision (bool): a bool value, whether to use high-precision version.

    Returns:
        tvm.tensor.Tensor of same type and shape as data.

    Supported Platforms:
        'Ascend', 'GPU'
    """
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    shape = [x.value for x in data.shape]
    utils.check_shape(shape)

    res = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(1, data.dtype) / data(*indice), name="res")

    # When product is mini, using Newtom iteration method to achieve higher precision.
    if product_is_mini() and high_precision:
        steps = 1
        for _ in range(steps):
            temp1 = data * res
            temp2 = temp1 * akg.tvm.const(-1, data.dtype)
            temp3 = temp2 + akg.tvm.const(2, data.dtype)
            res = temp3 * res

    return res
