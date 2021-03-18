# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function: sqrt"""
import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def sqrt(data):
    """
    Computes square root of x element-wise.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has same type and shape as data.
    """
    check_list = ["float16", "float32"]
    dtype = data.dtype
    if not dtype in check_list:
        raise RuntimeError("Sqrt cce only support %s while dtype is %s" % (
            ",".join(check_list), dtype))

    shape = [x.value for x in data.shape]
    vc_util.check_shape(shape)

    res = akg.topi.sqrt(data)
    return res
