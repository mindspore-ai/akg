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

"""operator dsl function: mul"""
import akg.topi
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def mul(l_input, r_input, target=utils.CCE):
    """
    Calculate x * y element-wise.

    Note:
        mul supports broadcasting.

    Args:
        l_input (tvm.tensor.Tensor): Tensor of type float16, float32.
        r_input (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has the same type as l_input and r_input.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    utils.ops_dtype_check([l_input.dtype, r_input.dtype], utils.DtypeForDavinci.ALL_FLOAT)

    shape1 = [x.value for x in l_input.shape]
    shape2 = [x.value for x in r_input.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)
    utils.auto_broadcast_check(shape1, shape2)
    utils.elemwise_dtype_check(l_input.dtype, r_input.dtype)
    output = akg.topi.multiply(l_input, r_input)

    return output
