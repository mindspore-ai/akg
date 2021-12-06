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

"""operator dsl function:greater"""

import akg
import akg.lang.ascend
import akg.tvm
from akg.utils.dsl_create import produce_shapes
import akg.utils as utils


def greater(data1, data2, target="cce"):
    # check shapes
    shape1 = [x.value for x in data1.shape]
    shape2 = [x.value for x in data2.shape]
    shapes = [shape1, shape2]
    for i in range(len(shapes)):
        utils.check_shape(shapes[i])

    # check types
    check_list = ["float16"]
    dtype = data1.dtype
    if not (dtype.lower() in check_list):
        raise RuntimeError("greater only support %s while dtype is %s" % (",".join(check_list), dtype))
    dtype = data2.dtype
    if not (dtype.lower() in check_list):
        raise RuntimeError("greater only support %s while dtype is %s" % (",".join(check_list), dtype))

    res = akg.topi.greater(data1, data2)
    return res
