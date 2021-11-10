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

"""operator dsl function: floormod"""

import akg.tvm
import akg
import akg.utils as utils
from akg.utils.kernel_exec import debug_mode



def floormod(shape, dtype, kernel_name, attrs, target="cce"):
    """
    Compute element-wise remainder of division.
    \f$res=a - floor(a/b) * b\f$

    Args:
         shape (list): a list has any nums.
         dtype (str): parameters' type.
         kernel_name (str): a str about kernel_name.
         attrs (str): Default None.
    Returns:
            tvm.tensor.Tensor, shape and dtype are input params.
    """

    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])
    utils.check_shape(shape)

    a = akg.tvm.placeholder(shape=shape, name="a", dtype=dtype)
    b = akg.tvm.placeholder(shape=shape, name="b", dtype=dtype)

    # res = a - floor(a/b) * b
    # Newton's Method for VREC
    para = akg.lang.ascend.vrec(b)
    for _ in range(3):
        tmp1 = akg.lang.ascend.vmul(b, para)
        tmp2 = akg.lang.ascend.vmuls(tmp1, -1)
        tmp3 = akg.lang.ascend.vadds(tmp2, 2)
        para = akg.lang.ascend.vmul(tmp3, para)

    c = akg.lang.ascend.vmul(a, para)
    d = akg.lang.ascend.floor(c)
    e = akg.lang.ascend.vmul(d, b)
    res = akg.lang.ascend.vsub(a, e)

    s = akg.tvm.create_schedule(res.op)

    with akg.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [a, b, res], "cce", name=kernel_name, attrs=attrs, polyhedral=True)
        return mod
