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

"""maxpool with argmax"""
from akg.ops.nn import maxpool


def MaxPoolWithArgmax(x, pad_mode="valid", window=1, pad=0, stride=1):
    """maxpool with argmax"""
    window = int(window)
    stride = int(stride)
    pad = int(pad)
    kernel = (window, window)
    stride_ = (stride, stride)
    strategy = pad_mode.upper()
    if pad_mode.upper() == "PAD":
        strategy = [pad, pad, pad, pad]
    return maxpool.maxpool_with_argmax(x, kernel, stride_, strategy)
