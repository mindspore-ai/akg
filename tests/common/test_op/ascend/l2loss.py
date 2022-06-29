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

"""l2loss"""
import akg.tvm
import akg.topi
import akg
import akg.lang.ascend
from akg.ops.math import sum
import akg.utils as utils


def l2loss(data, target="cce"):
    dtype = data.dtype

    check_list = ["float16", "float32"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("tile_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    utils.check_shape(data.shape)

    orig_dtype = dtype
    if dtype.lower() == "float16":
        dtype = "float32"
        data = akg.topi.cast(data, dtype)

    # code has bug
    #shape, axis = simplify_axis_shape(shape, range(len(shape)))

    coeff_sqrt = akg.tvm.const(1.0 / (2 ** (0.5)), dtype=dtype)

    res = akg.lang.ascend.vmuls(data, coeff_sqrt)
    res = akg.lang.ascend.vmul(res, res)
    res = sum(res, target=target)

    if dtype != orig_dtype:
        res = akg.topi.cast(res, orig_dtype)

    return res
