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

"""conv_backprop_input"""
from akg.ops.nn import conv_backprop_filter

def Conv2DBackpropFilter(out_backprop, input, input_shape, filter_sizes, pad_list, stride=1, dilation=1):
    """back propagation of 2d convolution on filter"""
    if len(pad_list) != 4:
        raise IndexError("Length of pad must be equal 4")
    pad_ = pad_list
    data = []
    data.append(out_backprop)
    data.append(input)
    fmap_shape = input_shape
    filter_shape = filter_sizes
    stride_ = [stride, stride]
    dilation_ = [dilation, dilation]

    return conv_backprop_filter.conv_backprop_filter(data, fmap_shape, filter_shape, pad_, stride_, dilation_)
