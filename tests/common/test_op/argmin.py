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

"""operator dsl function: argmin"""
import akg
from akg.ops.math import argmin_argmax_common as cn
from akg.utils import validation_check as vc_util

@vc_util.check_input_type(akg.tvm.tensor.Tensor, int)
def argmin(data, axis):
    """
    Calculate argmin value on specific axis.

    Args:
        data (tvm.tensor.Tensor): Target data.
        axis (int): A int number for which argmax calculate on.

    Returns:
        tvm.tensor.Tensor. As minimum number indexes.
    """
    res, attrs = cn.common(data, axis, "min")
    return res, attrs
