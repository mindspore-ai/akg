#!/usr/bin/env python3
# coding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

from swft.core import Tensor, name_tensor
from swft.intrinsic import *
from swft.utils import *


@name_tensor
def vgather(src, indices, axis=0, batchdims=0):
    attrs = {"mem_type": "UB", "format": None,
             "axis": [axis], "batchdims": [batchdims]}
    out_size = gather_shape_infer(src.shape, indices.shape, attrs)
    dst = Tensor("UB", src.dtype, out_size, src.format, src.multi_core)
    Vgather(src, indices, dst, attrs)()
    return dst
