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

"""operator dsl function: scale"""

import akg.tvm
import akg.topi
from akg.ops.math import Cast
import akg.utils as utils


def scale(input_data, scale_data, target="cce"):
    """
    Computes scaled input_data, res = input_data * scale_data

    Args:
        input_data(akg.tvm.Tensor): Tensor of type float16, float32, int8, uint8, int32.
        scale_data(akg.tvm.Tensor): Tensor of same type as input_data, if shape(scale_data) != shape(input_data),
                                    the shape of scale_data will broadcast to shape(input_data).

    Returns:
        akg.tvm.Tensor of same type and shape as input_data
    """

    # check shape
    input_data_shape = [x.value for x in input_data.shape]
    scale_shape = [x.value for x in scale_data.shape]
    utils.check_shape(input_data_shape)
    utils.check_shape(scale_shape)

    # check type
    check_list = ["float16", "float32", "int8", "uint8", "int32"]
    dtype = input_data.dtype
    if not dtype in check_list:
        raise TypeError("scale_data operator only supports %s while dtype is %s" % (",".join(check_list), dtype))
    if scale_data.dtype != dtype:
        raise TypeError("type(input_data) is %s, type(scale_data) is %d, which is inconsistent" % (
            dtype, scale_data.dtype))

    orig_dtype = dtype
    if dtype == "int8" or dtype == "uint8":
        dtype = "float16"
    if dtype == "int32":
        dtype = "float32"
    if dtype != orig_dtype:
        input_data = Cast(input_data, dtype, target=utils.CCE)
        scale_data = Cast(scale_data, dtype, target=utils.CCE)

    if scale_shape != input_data_shape:
        scale_data = akg.topi.broadcast_to(scale_data, input_data_shape)

    res = akg.tvm.compute(input_data_shape, lambda *indice: input_data(*indice) * scale_data(*indice), name="res")

    if res.dtype != orig_dtype:
        res = Cast(res, orig_dtype, target=utils.CCE)

    return res


def scale_bias(input_data, scale_data, bias_data, target="cce"):
    """
    Adds bias_data on scaled input_data, res = input_data * scale_data + bias_data

    Args:
        input_data(akg.tvm.Tensor): Tensor of type float16, float32, int8, uint8, int32.
        scale_data(akg.tvm.Tensor): Tensor of same type as input_data, if shape(scale_data) != shape(input_data),
                                    the shape of scale_data will broadcast to shape(input_data).
        bias_data(akg.tvm.Tensor): Tensor of same type as input_data, if shape(bias_data) != shape(input_data),
                                   the shape of bias_data will broadcast to shape(input_data).

    Returns:
        akg.tvm.Tensor of same type and shape as input_data.
    """

    # check shape
    input_data_shape = [x.value for x in input_data.shape]
    bias_shape = [x.value for x in bias_data.shape]
    utils.check_shape(bias_shape)

    # check type
    if bias_data.dtype != input_data.dtype:
        raise RuntimeError("type(input_data) is %s, type(bias_data) is %d, which is inconsistent" % (
            input_data.dtype, bias_data.dtype))

    scale_input_data = scale(input_data, scale_data)

    dtype = bias_data.dtype
    orig_dtype = dtype
    if dtype == "int8" or dtype == "uint8":
        dtype = "float16"
    if dtype == "int32":
        dtype = "float32"
    if dtype != orig_dtype:
        scale_input_data = Cast(scale_input_data, dtype, target=utils.CCE)
        bias_data = Cast(bias_data, dtype, target=utils.CCE)

    if bias_shape != input_data_shape:
        bias_data = akg.topi.broadcast_to(bias_data, input_data_shape)

    res = akg.tvm.compute(input_data_shape, lambda *indice: scale_input_data(*indice) + bias_data(*indice), name="res_bias")

    if res.dtype != orig_dtype:
        res = Cast(res, orig_dtype, target=utils.CCE)

    return res
