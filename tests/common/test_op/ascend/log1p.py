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

"""operator dsl function: log1p"""

import akg.tvm
from akg.utils import custom_tiling as ct_util
import akg.utils as utils


log1p_set_dim_map = {
    str(([[3, 2, 30588]], ["float32"])): ((1, 1), (1, 1), (7647, 1))
}


def npy_log1p(x, dtype):
    c1 = akg.tvm.const(1.0, dtype=dtype)
    y = x + c1
    z = y - c1
    t = x / z
    return akg.tvm.log(y) * t


def gsl_log1p(x, dtype):
    c1 = akg.tvm.const(1.0, dtype=dtype)
    y = x + c1
    z = y - c1
    return akg.tvm.log(y) - (z - x) / y   # cancels errors with IEEE arithmetic.


@ct_util.reg_set_dim_func(log1p_set_dim_map)
def log1p(data):
    """
    Computes natural logarithm of (1 + data).

    Args:
        data(akg.tvm.Tensor): Tensor of float16 or float32.

    Returns:
        akg.tvm.Tensor of same shape and type as input tensor data.
    """

    check_list = ["float16", "float32"]
    if not data.dtype in check_list:
        raise RuntimeError("tile_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    utils.check_shape(data.shape)

    # numpy implement
    func = lambda *i: npy_log1p(data(*i), data.dtype)

    res = akg.tvm.compute(data.shape, func, name="log1p")

    return res
