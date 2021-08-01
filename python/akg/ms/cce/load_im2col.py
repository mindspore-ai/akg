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

"""load_im2col"""
from akg.ops.nn import load_im2col
from akg.utils.format_transform import get_shape
import math


def LoadIm2Col(x, ksizes, strides):
    """load_im2col"""
    bs, c1, h, w, c0 = get_shape(x)
    stride_h, stride_w = strides
    k_w, k_h = ksizes
    dilation_h = 1
    dilation_w = 1
    h_out = math.ceil(h / stride_h)
    w_out = math.ceil(w / stride_w)
    pad_needed_h = max(0, (h_out - 1) * stride_h + dilation_h * (k_h - 1) + 1 - h)
    pad_top = math.floor(pad_needed_h / 2)
    pad_bottom = pad_needed_h - pad_top
    pad_needed_w = max(0, (w_out - 1) * stride_w + dilation_w * (k_w - 1) + 1 - w)
    pad_left = math.floor(pad_needed_w / 2)
    pad_right = pad_needed_w - pad_left
    pad_list = [pad_top, pad_bottom, pad_left, pad_right]
    return load_im2col.load_im2col(x, ksizes, strides, pad=pad_list)
