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

"""operator dsl function: ones_like"""

import akg.tvm
import akg.utils as utils
from akg.ops.math import cast
from akg.utils.format_transform import get_shape

@utils.check_input_type(akg.tvm.tensor.Tensor)
def ones_like(input):
    """
    Generate an array of ones.

    Args:
        input (tvm.tensor.Tensor): Tensor,Should be of type float16, float32, int32, uint8, int8.

    Returns:
        tvm.tensor.Tensor with the same type and shape as input.
    """
    dtype = input.dtype
    shape = get_shape(input)
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.ALL_TYPES])
    utils.check_shape(shape)
    res = akg.tvm.compute(shape, lambda *i: akg.tvm.const(1, "float16"), name="res", attrs={'no_inline': 1})
    res = cast(res, dtype, target=utils.CCE)
    return res
