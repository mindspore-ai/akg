# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: sub"""
import akg.topi
import akg.tvm
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def sub(data1, data2, target=utils.CCE):
    """
    Computes data1 - data2 elementwise, broadcast is supported.

    Args:
        data1 (tvm.tensor.Tensor): Tensor of type float16, float32, int32.
        data2 (tvm.tensor.Tensor): Tensor of same type as data1, if shape(data2) != shape(data1), broadcast will happen.

    Returns:
        tvm.tensor.Tensor, subtracted result, with same type as input tensors and broadcasted shape of data1 and data2.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        utils.ops_dtype_check([data1.dtype, data2.dtype],
                        [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])
    utils.elemwise_dtype_check(data1.dtype, data2.dtype)

    utils.check_shape(data1.shape)
    utils.check_shape(data2.shape)
    utils.auto_broadcast_check(data1.shape, data2.shape)

    res = akg.topi.subtract(data1, data2)

    return res
