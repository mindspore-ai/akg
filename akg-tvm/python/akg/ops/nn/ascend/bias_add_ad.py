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

"""operator dsl function: bias_add_ad"""

import akg.tvm
import akg
import akg.utils as utils
from akg.ops.nn.ascend.bias_add import bias_add


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), str, (str, type(None)))
def bias_add_ad(head, input_shape, data_format, target=utils.CCE):
    """
    Compute gradient for bias_add operator using automatic differentiate.

    Args:
        head (tvm.tensor.Tensor): Input tensor.
        input_shape (Union[list, tuple]): Input shape of head.
        data_format (str): Data format of input tensors.

    Returns:
        tvm.tensor.Tensor of same shape and type as head.

    Supported Platforms:
        'Ascend'
    """

    check_list = ["NHWC", "NC1HWC0", "DefaultFormat"]
    if data_format not in check_list:
        raise RuntimeError("bias_add_grad only support %s while dataformat is %s" %
                           (",".join(check_list), data_format))
    utils.check_shape(head.shape)
    shape1 = [x.value for x in head.shape]
    utils.davinci_format_check(shape1, data_format)
    a = akg.tvm.placeholder(head.shape, head.dtype, "A")
    if data_format == "NC1HWC0":
        bias_shape = (1, head.shape[1], 1, 1, head.shape[4])
        b = akg.tvm.placeholder(bias_shape, head.dtype, "B")
    elif data_format == "NHWC":
        bias_shape = (input_shape[-1],)
        b = akg.tvm.placeholder(bias_shape, head.dtype, "B")
    else:
        bias_shape = (input_shape[1],)
        b = akg.tvm.placeholder(bias_shape, head.dtype, "B")
    c = bias_add(a, b, data_format)

    jacs = list(akg.differentiate(c, [b], head))
    attrs = {}
    return jacs[0], attrs
