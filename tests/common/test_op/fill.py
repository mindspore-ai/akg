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

"""operator dsl function:fill"""


import akg.tvm
from akg.utils import validation_check as vc_util


def fill(shape, value, dtype):
    
    vc_util.ops_dtype_check(dtype, [vc_util.DtypeForDavinci.FLOAT16, vc_util.DtypeForDavinci.INT32])
    vc_util.check_shape(shape)

    A = akg.tvm.const(value, dtype)
    res = akg.tvm.compute(shape, lambda *i: A, name="fill")
    return res
