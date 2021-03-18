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

"""operator dsl function: log_ad"""

import akg
import akg.tvm
from akg.ops.math import log
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def log_ad(head, in_data):
    """
    Compute gradient of log operator using automatic differentiate.

    Args:
        head (tvm.tensor.Tensor): Tensor of type float16, float32.
        in_data (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor has the same shape as input.
    """

    # check head's validation.
    vc_util.check_shape(head.shape)
    vc_util.ops_dtype_check(head.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    b = log.log(in_data)
    jacs = list(akg.differentiate(b, [in_data], head))
    return jacs[0]
