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

"""operator dsl function:bitwise_not"""

import akg.tvm
import akg
from akg.utils import validation_check as vc_util
from akg.ops.math.cast import cast

@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def bitwise_not(data):
    """
    Bitwise-not.

    Args:
        data (tvm.tensor.Tensor): Input data of type int8 or int32.

    Returns:
        tvm.tensor.Tensor, Bitwise-not result.
    """
    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.ALL_INT)
    vc_util.check_shape(data.shape)

    one = akg.tvm.const(1, dtype=data.dtype)
    minus_one = akg.tvm.const(-1, dtype=data.dtype)
    add_one = akg.lang.cce.vadds(data, one)
    multiply_one = akg.lang.cce.vmuls(add_one, minus_one)
    res = cast(multiply_one, data.dtype)
    return res
