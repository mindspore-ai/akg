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

"""operator dsl function:square"""

import akg
from akg.utils import validation_check as vc_util


def square(data):
    """
    Compute square.

    Args:
        data: Tensor.

    Return:
        Tensor, has the same shape and type as data.
    """
    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.ALL_TYPES)
    shape = [x.value for x in data.shape]
    vc_util.check_shape(shape)

    res = akg.lang.cce.vmul(data, data)
    return res
