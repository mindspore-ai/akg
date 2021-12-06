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

"""operator dsl function:tanh"""
import akg
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini
from .rec_positive import RecPositive

@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def Tanh(in_data, target=utils.CCE):
    """
    Compute tanh function. This version is able to avoid exp(x) overflow when x is large.

    ..math:`res = sign(in_data) * (1 - exp(-2*abs(in_data))) / (1 + exp(-2*abs(in_data)))`

    Args:
        in_data (tvm.tensor.Tensor): input tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as in_data.
    
    Supported Platforms:
        'Ascend'
    """

    utils.check_shape(in_data.shape)

    dtype = in_data.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    ori_dtype = dtype
    in_data_compute = in_data
    if ori_dtype == "float32" and product_is_mini():
        in_data_compute = akg.tvm.compute(in_data.shape, lambda *indice: in_data(* \
                                          indice).astype("float16"), name='type_cast')
        dtype = 'float16'

    in_data_abs = akg.lang.ascend.vabs(in_data_compute)
    exponent = akg.lang.ascend.vmuls(in_data_abs, akg.tvm.const(-2, dtype))
    exp_value = akg.lang.ascend.vexp(exponent)

    exp_value_add_one = akg.lang.ascend.vadds(exp_value, akg.tvm.const(1, dtype))
    one_sub_exp_value = akg.topi.subtract(akg.tvm.const(1, dtype), exp_value)
    exp_value_add_one_rec = RecPositive(exp_value_add_one, target)
    tanh_value_pos = akg.topi.multiply(one_sub_exp_value, exp_value_add_one_rec)
    output_shape = in_data_compute.shape
    sign = akg.tvm.compute(output_shape,
                           lambda *indice:
                           akg.tvm.expr.Select(in_data_compute(*indice) < akg.tvm.const(0, dtype),
                                               akg.tvm.const(-1, dtype), akg.tvm.const(1, dtype)))

    tanh_value = akg.topi.multiply(sign, tanh_value_pos)
    if ori_dtype == "float32" and product_is_mini():
        tanh_value = akg.tvm.compute(tanh_value.shape,
                                     lambda *indice: tanh_value(*indice).astype("float32"),
                                     name='res')

    return tanh_value
