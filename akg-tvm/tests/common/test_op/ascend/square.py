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

"""operator dsl function:square"""

import akg
import akg.utils as utils


def square(data, target="cce"):
    """
    Compute square.

    Args:
        data: Tensor.

    Return:
        Tensor, has the same shape and type as data.
    """
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_TYPES)
    shape = [x.value for x in data.shape]
    utils.check_shape(shape)

    res = akg.lang.ascend.vmul(data, data)
    return res
