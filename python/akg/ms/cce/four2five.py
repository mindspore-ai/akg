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

"""four2five"""
from akg.ops.array import four2five
from akg.ms.utils import DEFAULT

def Four2Five(x, data_format=None, dst_type="float16"):
    """from 4d(NCHW) to 5d(NC1HWC0)"""
    if data_format is None:
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    if data_format == DEFAULT:
        data_format = "NCHW"
    return four2five.four2five(x, data_format, dst_type)
