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

"""operator dsl function: expand_dims"""
import akg.topi
import akg.tvm
import akg.utils as utils

@utils.check_input_type(akg.tvm.tensor.Tensor, int, (str, type(None)))
def ExpandDims(data, axis, target=utils.CCE):
    """
    Computes data1 elementwise.

    Args:
        data1 (tvm.tensor.Tensor): Tensor.
        axis (int): axis.

    Returns:
        tvm.tensor.Tensor, expand the dimension of data1.
    
    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    utils.check_shape(data.shape)
    if target == utils.CCE:
        utils.ops_dtype_check(data.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])
        res = akg.topi.expand_dims(data, axis, 1)
    else:
        res = akg.topi.expand_dims(data, axis)

    return res
