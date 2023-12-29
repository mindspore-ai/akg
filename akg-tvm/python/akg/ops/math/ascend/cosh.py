# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function:cosh"""
import akg
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.utils.kernel_exec import product_is_mini


def get_attrs():
    """get attrs."""
    attrs = {
        "enable_feature_library": True
    }
    return attrs


def cosh_call(x):
    """Compute cosh by the call method."""
    dtype = x.dtype
    shape = get_shape(x)
    # in order to get the precise calcuate result
    if product_is_mini() and dtype == "float32":
        x = akg.lang.ascend.cast_to(x, "float16")

    res = akg.tvm.compute(shape, lambda *indice: akg.lang.ascend.cosh(x(*indice)), name="res")

    if product_is_mini() and dtype == "float32":
        res = akg.lang.ascend.cast_to(res, "float32")

    return res, get_attrs()


def cosh_compute(data):
    """compute cosh."""
    dtype = data.dtype
    if dtype is not "float32":
        data = akg.lang.ascend.cast_to(data, "float32")

    neg_data = akg.lang.ascend.vmuls(data, akg.tvm.const(-1, "float32"))
    data_exp = akg.lang.ascend.vexp(data)
    neg_data_exp = akg.lang.ascend.vexp(neg_data)
    sum_exp = akg.lang.ascend.vadd(data_exp, neg_data_exp)
    res = akg.lang.ascend.vmuls(sum_exp, akg.tvm.const(0.5, "float32"))
    if dtype is not "float32":
        res = akg.lang.ascend.cast_to(res, dtype)
    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def cosh(data, target=utils.CCE):
    """
    cosh op for input tensor.

    ..math:`y = (e^(x)+e^(-x))/2`

    Args:
        data (tvm.tensor.Tensor): tensor with type float16 or float32.

    Returns:
        tvm.tensor.Tensor.

    Supported Platforms:
        'Ascend'
    """
    dtype = data.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)

    utils.check_shape(data.shape)

    return cosh_call(data)
