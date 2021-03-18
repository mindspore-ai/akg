# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function: softplus"""
import akg
from akg import tvm
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util, kernel_exec as utils


# define a scalar, value = 1
SCALAR_ONE = 1


def softplus_compute(data):
    """compute for softplus"""
    dtype = data.dtype

    need_cast_back = False
    if utils.product_is_mini():
        if dtype == "float32":
            data = akg.lang.cce.cast_to(data, "float16")
            need_cast_back = True
    else:
        if dtype == "float16":
            data = akg.lang.cce.cast_to(data, "float32")
            need_cast_back = True

    data_exp = akg.lang.cce.vexp(data)
    data_add = akg.lang.cce.vadds(data_exp, SCALAR_ONE)
    res = akg.lang.cce.vlog(data_add)

    if need_cast_back:
        res = akg.lang.cce.cast_to(res, dtype)

    return res


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def softplus(data):
    """
    Compute for softplus.

    .. math::
        y = log\\left(e^x + 1\\right)

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16 or float32.

    Returns:
        tvm.tensor.Tensor with same shape and dtype as inputs.
    """
    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    vc_util.check_shape(data.shape)

    return softplus_compute(data)
