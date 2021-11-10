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

"""operator dsl function:square_difference"""
import akg.topi
import akg.tvm
import akg
import akg.lang.ascend
import akg.utils as utils


def square_difference(A, B, target="cce"):
    shape1 = [x.value for x in A.shape]
    shape2 = [x.value for x in B.shape]
    dtype = A.dtype

    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(shape1)
    utils.check_shape(shape2)

    Bp = akg.lang.ascend.vmuls(B, akg.tvm.const(-1, dtype))
    C = akg.topi.add(A, Bp)
    res = akg.topi.multiply(C, C)

    return res
