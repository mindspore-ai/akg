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

"""operator dsl function: relu_ad"""

import akg
import akg.tvm
import akg.utils as utils
from akg.utils import custom_tiling as ct_util
from akg.ops.nn.ascend.relu import Relu
from akg.dim import DIM

relu_ad_set_dim_map = {
}


def relu_ad_set_dim_func(head, a):
    """set dim info"""
    key = []
    key.append(tuple(a.shape))
    key.append(a.dtype)
    hash_key = str(tuple(key))

    if hash_key in relu_ad_set_dim_map.keys():
        return ct_util.set_dims(relu_ad_set_dim_map[hash_key]), hash_key
    return "", hash_key


@ct_util.reg_set_dim_func(relu_ad_set_dim_func)
@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def ReluAd(head, a, target=utils.CCE):
    """
    Compute gradient of relu operator using automatic differentiate.

    Args:
        head (tvm.tensor.Tensor): Tensor of type float16, float32, int8, uint8, int32.
        a (tvm.tensor.Tensor): Tensor of type float16, float32, int8, uint8, int32.

    Returns:
        tvm.tensor.Tensor with the same shape as input.
    
    Supported Platforms:
        'Ascend'
    """
    dim_info, _ = relu_ad_set_dim_func(head, a)
    attrs = {DIM: dim_info}

    b = Relu(a)
    jacs = list(akg.differentiate(b, [a], head))
    return jacs[0], attrs
