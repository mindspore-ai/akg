# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: sinh"""
import akg
import akg.utils as utils
from akg import topi, tvm
from akg.utils.format_transform import get_shape
from ..exp import exp


def sinh_compute(x):
    """Compute sinh."""
    dtype = x.dtype
    # in order to get the precise calcuate result
    if dtype == "float16":
        x = topi.cast(x, "float32")

    data_exp = exp(x, utils.CCE)
    negative_data = topi.multiply(x, -1)
    negative_data_exp = exp(negative_data, utils.CCE)
    data_exp_sub = topi.subtract(data_exp, negative_data_exp)

    res = topi.multiply(data_exp_sub, tvm.const(0.5, "float32"))
    if dtype == "float16":
        res = topi.cast(res, "float16")

    return res


def get_attrs():
    """get attrs."""
    attrs = {
        "enable_feature_library": True
    }
    return attrs


def sinh_call(x):
    """Compute sinh."""
    dtype = x.dtype
    shape = get_shape(x)
    # in order to get the precise calcuate result
    if dtype == "float16":
        x = akg.lang.ascend.cast_to(x, "float32")

    res = akg.tvm.compute(shape, lambda *indice: akg.lang.ascend.sinh(x(*indice)), name="res")

    if dtype == "float16":
        res = akg.lang.ascend.cast_to(res, "float16")

    return res, get_attrs()


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def Sinh(x, target=utils.CCE):
    """
    Computes hyperbolic sine of `x` element-wise.

    .. math::
        sinh(x) = \\frac{e^x - e^{-x}}{2}

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.

    Rerurns:
        tvm.tensor.Tensor of same type and shape as in_data.
    
    Supported Platforms:
        'Ascend'
    """
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(x.shape)

    use_call = True
    if use_call:
        return sinh_call(x)

    return sinh_compute(x)
