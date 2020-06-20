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

"""operator dsl function:expm1"""

import akg
from akg.ops.math.exp import exp
import akg.utils.validation_check as vc_util


def expm1(data):
    """
    Calculate exp(x) - 1.

    Calculate \f$e^{x}-1\f$, where x is the input tensor and e is Euler's number.

    Args:
        data: Tensor.

    Returns:
        Tensor, has the same type and shape as data.
    """

    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    output = akg.lang.cce.vadds(exp(data), -1)

    return output
