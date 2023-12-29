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

"""operator dsl function: addn"""
import akg
import akg.topi
import akg.utils as utils


@utils.check_input_type(((list, tuple), akg.tvm.tensor.Tensor), (str, type(None)))
def addn(data, target=utils.CCE):
    """
    Compute sum of all elements in tensor.

    Args:
        data (tvm.tensor.Tensor): Tensor of of type float16, float32.

    Returns:
        tvm.tensor.Tensor, compute result, get all elements' sum.
    
    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    # check types
    dtype = data[0].dtype
    if target == utils.CCE:
        utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)

    res = data[0]
    for i in range(1, len(data)):
        utils.elemwise_dtype_check(res.dtype, data[i].dtype)
        utils.elemwise_shape_check(res.shape, data[i].shape)
    res = akg.topi.elemwise_sum(data)

    return res