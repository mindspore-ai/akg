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

"""operator dsl function: argmax"""
import akg
import akg.utils as utils


from .argmin_argmax_common import common


@utils.check_input_type(akg.tvm.tensor.Tensor, int, (str, type(None)))
def argmax(data, axis):
    """
    Calculate argmax value on specific axis.

    Args:
        data (tvm.tensor.Tensor): Input tensor.
        axis (int): Specifies which axis to reduce.

    Returns:
        Tensor as maximum number indexes.
    
    Supported Platforms:
        'Ascend'
    """
    res, attrs = common(data, axis, "max")
    return res, attrs
