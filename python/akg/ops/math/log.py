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

"""operator dsl function: log"""
import akg.tvm
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util

@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def log(data):
    """
    Compute natural logarithm of x element-wise.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32, int8, uint8, int32.

    Returns:
        tvm.tensor.Tensor of same type and shape as data
    """

    in_data = data
    dtype = in_data.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    if dtype == "float32" and utils.product_is_mini():
        in_data = akg.tvm.compute(in_data.shape, lambda *indice: in_data(*indice).astype("float16"), name='type_cast')

    output = akg.tvm.compute(in_data.shape, lambda *index: akg.tvm.log(in_data(*index)), name='log')

    if dtype == "float32" and utils.product_is_mini():
        output = akg.tvm.compute(in_data.shape, lambda *indice: output(*indice).astype("float32"), name='res')

    return output
