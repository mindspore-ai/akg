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

"""operator dsl function:greater_equal"""
import akg.tvm
import akg
import akg.lang.cce
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import produce_shapes


def greater_equal(data1, data2):
    # check shapes
    shape1 = [x.value for x in data1.shape]
    shape2 = [x.value for x in data2.shape]
    shapes = [shape1, shape2]
    for i in range(len(shapes)):
        vc_util.check_shape(shapes[i])

    # check types
    dtype = data1.dtype
    dtype2 = data2.dtype
    vc_util.elemwise_dtype_check(dtype, dtype2)
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.FLOAT16)

    res = akg.topi.greater_equal(data1, data2)
    return res
