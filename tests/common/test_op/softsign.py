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
from akg.ops.math.reciprocal import reciprocal
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util, kernel_exec as utils


# define a scalar, value = 1
SCALAR_ONE = 1


def softsign_compute(input_features):
    """ompute for softsign"""
    dtype = input_features.dtype
    if dtype == "float16":
        input_features = akg.lang.cce.cast_to(input_features, "float32")

    data_abs = akg.lang.cce.vabs(input_features)
    data_add = akg.lang.cce.vadds(data_abs, SCALAR_ONE)
    data_rec = reciprocal(data_add)
    res = akg.lang.cce.vmul(input_features, data_rec)

    if dtype == "float16":
        res = akg.lang.cce.cast_to(res, "float16")

    return res


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def softsign(data):
    """
    Computes for softsign.

    .. math::
        y = \\frac{x}{\\left|x\\right| + 1}

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16 or float32.

    Returns:
        tvm.tensor.Tensor with same shape and dtype as inputs.
    """
    vc_util.check_shape(data.shape)
    vc_util.tensor_max_size_check(data)
    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    return softsign_compute(data)
