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

"""operator dsl function:bitwise_not"""

import akg.tvm
import akg
import akg.utils as utils
from akg.ops.math import Cast

@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def bitwise_not(data, target=utils.CCE):
    """
    Bitwise-not.

    Args:
        data (tvm.tensor.Tensor): Input data of type int8 or int32.

    Returns:
        tvm.tensor.Tensor, Bitwise-not result.
    """
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_INT)
    utils.check_shape(data.shape)

    one = akg.tvm.const(1, dtype=data.dtype)
    minus_one = akg.tvm.const(-1, dtype=data.dtype)
    add_one = akg.lang.ascend.vadds(data, one)
    multiply_one = akg.lang.ascend.vmuls(add_one, minus_one)
    res = Cast(multiply_one, data.dtype, target=target)
    return res
