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

"""operator dsl function:l2normalize"""

import akg

from akg.utils import kernel_exec as utils
from akg.ops.math.sum import sum_value
from akg.ops.math.rsqrt import rsqrt
from akg.utils import validation_check as vc_util


def l2normalize(data):
    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    vc_util.check_shape(data.shape)
    square_res = akg.lang.cce.vmul(data, data)
    reduce_sum, _ = sum_value(square_res, -1, keepdims=True)
    one_of_square = rsqrt(reduce_sum)
    broad_cast = akg.lang.cce.broadcast(one_of_square, data.shape)
    res = akg.lang.cce.vmul(data, broad_cast)
    attrs = {"pragma_modshift": 1}
    return res, attrs
