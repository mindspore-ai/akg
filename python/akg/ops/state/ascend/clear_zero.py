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

"""operator dsl function: clear_zero"""
import akg.tvm
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils

@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def ClearZero(data, target=utils.CCE):
    """
    Sets all elements in tensor to zero.

    Args:xiasn
         data (tvm.tensor.Tensor): Tensor needs to be cleared to zero.

    Returns:
         out: tvm.tensor.Tensor will all elements with value zero.
         attrs: dict.
    """

    shape = [x for x in data.shape]

    zero = akg.tvm.const(0, data.dtype)
    out = akg.tvm.compute(shape, lambda *i: zero, "out")
    out, binds_info = TensorUtils.inplace_set(data, out)
    attrs = {utils.BINDS: binds_info}
    return out, attrs
