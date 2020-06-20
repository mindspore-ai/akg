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

"""maxpool grad with argmax"""
from akg.ops.nn import maxpool_grad_with_argmax
from akg.utils.format_transform import get_shape_from_tensor


def MaxPoolGradWithArgmax(x, argmax, grad, pad_mode="valid", window=1, pad=0, stride=1):
    """maxpool grad with argmax"""
    kernel = (window, window)
    stride_ = (stride, stride)
    strategy = pad_mode.upper()
    if pad_mode.upper() == "PAD":
        strategy = [pad, pad, pad, pad]
    input_shape = get_shape_from_tensor(x)
    return maxpool_grad_with_argmax.maxpool_grad_with_argmax(head=grad, mask=argmax, shape=input_shape, kernel=kernel,
                                                             stride=stride_, pad=strategy)
