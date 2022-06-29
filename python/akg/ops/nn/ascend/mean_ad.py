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

"""operator dsl function: mean_ad"""
import akg.tvm
import akg
from akg.ops.math.ascend.mean import mean
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), (list, tuple, int), bool, (str, type(None)))
def MeanAd(head, input_shape, axis, keepdims, target=utils.CCE):
    """
    Compute gradient of mean operator using automatic differentiate.

    Args:
        head (tvm.tensor.Tensor): Input tensor.
        input_shape (Union[list, tuple]): Shape of input tensor of mean operator.
        axis (Union[list, tuple, int]): Specifies which axis to reduce.
        keepdims (bool): Keep the reduced axis with length 1 if keepdims is true.

    Returns:
        tvm.tensor.Tensor.

    Supported Platforms:
        'Ascend'
    """
    a = akg.tvm.placeholder(input_shape, head.dtype, "A")
    b, _ = mean(a, axis, keepdims, target=target)
    jacs = list(akg.differentiate(b, [a], head))
    return jacs[0]
