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

"""operator dsl function:l2normalize"""

import akg
import akg.utils as utils
from akg.ops.math import Sum, rsqrt

def l2normalize(data, target=utils.CCE):
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(data.shape)
    square_res = akg.lang.ascend.vmul(data, data)
    reduce_sum = Sum(square_res, -1, keepdims=True, target=target)
    one_of_square = rsqrt(reduce_sum, target=target)
    broad_cast = akg.lang.ascend.broadcast(one_of_square, data.shape)
    res = akg.lang.ascend.vmul(data, broad_cast)
    attrs = {"pragma_modshift": 1}
    return res, attrs
