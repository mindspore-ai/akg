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

"""operator dsl function: relu6"""

import akg.tvm
from akg import lang
from akg.utils import validation_check as vc_util

def relu6(inputs):
    """
    Computes Rectified Linear 6: min(max(features, 0), 6).

    Args:
        inputs (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, which has same type and shape as input.
    """

    dtype = inputs.dtype
    check_list = ["float16", "float32"]
    if not dtype in check_list:
        raise RuntimeError("relu6 only support %s while dtype is %s" % (",".join(check_list), dtype))

    shape = inputs.shape
    vc_util.check_shape(shape)

    zero = lang.cce.broadcast(akg.tvm.const(0, dtype=dtype), shape)
    max_inputs = lang.cce.vmax(inputs, zero)
    six = lang.cce.broadcast(akg.tvm.const(6, dtype=dtype), shape)
    res = lang.cce.vmin(max_inputs, six)

    return res
