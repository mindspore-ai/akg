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

"""operator dsl function: logical or"""

import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util


def logical_or(input1, input2):

    dtype1 = input1.dtype
    dtype2 = input2.dtype
    vc_util.elemwise_dtype_check(dtype1, dtype2)
    vc_util.ops_dtype_check(dtype1, vc_util.DtypeForDavinci.BOOL)

    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    vc_util.check_shape(shape1)
    vc_util.check_shape(shape2)

    vc_util.elemwise_shape_check(shape1, shape2)
    res = akg.topi.logical_or(input1, input2)

    return res
