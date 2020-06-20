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

"""operator dsl function: less_equal"""

import akg
import akg.lang.cce
import akg.tvm

import akg.topi
from akg.ops.math.sub import sub
from akg.ops.math.cast import cast
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import produce_shapes


def less_equal(data1, data2):
    # check shapes
    shape1 = [x.value for x in data1.shape]
    vc_util.check_shape(shape1)
    shape2 = [x.value for x in data2.shape]
    vc_util.check_shape(shape2)

    # check types
    if data1.dtype != data2.dtype:
        raise TypeError("data1 is of type %s, data2 is of type %s, which are different." % (data1.dtype, data2.dtype))

    check_list = ["float16", "float32", "int32"]
    dtype = data1.dtype
    orig_dtype = data1.dtype
    if not dtype in check_list:
        raise TypeError("less_equal only support %s while dtype is %s" % (",".join(check_list), dtype))

    if utils.product_is_mini():
        if dtype is not "float16":
            dtype = "float16"
    else:
        if dtype not in ["float32", "float16"]:
            dtype = "float32"

    if orig_dtype == "float32" and dtype == "float16":
        data_sub = sub(data1, data2)
        data_sub = akg.topi.cast(data_sub, dtype)
        zero = akg.tvm.const(0.0, dtype)
        res = akg.topi.less_equal(data_sub, zero)
    else:
        data1 = akg.topi.cast(data1, dtype)
        data2 = akg.topi.cast(data2, dtype)
        res = akg.topi.less_equal(data1, data2)

    return res
