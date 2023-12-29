# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: maximum"""
import akg.topi as topi
import akg.tvm as tvm
import akg.utils as utils
from .cast import cast


@utils.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (str, type(None)))
def maximum(data1, data2, target=utils.CCE):
    """
    Take element-wise maximum of two tensors with auto-broadcasting.

    Args:
        data1: tvm.tensor.Tensor
        data2: tvm.tensor.Tensor

    Returns:
        tvm.tensor.Tensor of maximum of two tensors.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    shape1 = [x.value for x in data1.shape]
    shape2 = [x.value for x in data2.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)
    utils.auto_broadcast_check(shape1, shape2)
    utils.elemwise_dtype_check(data1.dtype, data2.dtype)

    dtype = data1.dtype
    need_cast = True if target == utils.CCE and dtype in ["int8", "uint8"] else False
    if need_cast:
        data1 = cast(data1, "float16")
        data2 = cast(data2, "float16")
    res = topi.maximum(data1, data2)
    if need_cast:
        res = cast(res, dtype)
    return res
