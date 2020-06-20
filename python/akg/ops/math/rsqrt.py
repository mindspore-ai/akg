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

"""operator dsl function: rsqrt"""

import akg.tvm
import akg.topi
from akg.utils import kernel_exec as utils
from akg.utils.validation_check import check_shape, ops_dtype_check, check_input_type, DtypeForDavinci
from akg.utils.format_transform import get_shape


@check_input_type(akg.tvm.tensor.Tensor)
def rsqrt(data):
    """
    Computes reciprocal of square root of x element-wise.

     :math:`y = \frac{1}{\\sqrt x} = x^{-\frac{1}{2}}`

    Note:
        In order to prevent loss of precision, the function uses exponential constant changes:
        :math:`y = [e^{lnx}]^{-\frac{1}{2}}`

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32

    Returns:
        tvm.tensor.Tensor, has same type and shape as data.
    """

    dtype = data.dtype

    shape = get_shape(data)
    ops_dtype_check(dtype, DtypeForDavinci.ALL_FLOAT)
    check_shape(shape)

    if not utils.product_is_mini():
        return akg.topi.rsqrt(data)

    is_needed_conv = (dtype == 'float32')

    data_ = data.astype('float16') if is_needed_conv else data
    power_num = akg.tvm.const(-0.5, dtype=('float16' if is_needed_conv else dtype))

    vlog_t = akg.tvm.compute(
        shape, lambda *indice: akg.tvm.log(data_(*indice)), name="vlog_t")
    vmuls_t = akg.tvm.compute(
        shape, lambda *indice: vlog_t(*indice) * power_num, name="vmuls_t")
    res = akg.tvm.compute(shape, lambda *indice: akg.tvm.exp(vmuls_t(*indice)), name="res")

    res = res.astype('float32') if is_needed_conv else res

    return res
