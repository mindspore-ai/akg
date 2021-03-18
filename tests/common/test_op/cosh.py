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

"""operator dsl function:cosh"""
import akg
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.utils import kernel_exec as utils

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
    if utils.product_is_mini() and dtype == "float32":
        x = akg.lang.cce.cast_to(x, "float16")

    res = akg.tvm.compute(shape, lambda *indice: akg.lang.cce.cosh(x(*indice)), name="res")

    if utils.product_is_mini() and dtype == "float32":
        res = akg.lang.cce.cast_to(res, "float32")

    return res, get_attrs()


def cosh_compute(data):
    """compute cosh."""
    dtype = data.dtype
    if dtype is not "float32":
        data = akg.lang.cce.cast_to(data, "float32")

    neg_data = akg.lang.cce.vmuls(data, akg.tvm.const(-1, "float32"))
    data_exp = akg.lang.cce.vexp(data)
    neg_data_exp = akg.lang.cce.vexp(neg_data)
    sum_exp = akg.lang.cce.vadd(data_exp, neg_data_exp)
    res = akg.lang.cce.vmuls(sum_exp, akg.tvm.const(0.5, "float32"))
    if dtype is not "float32":
        res = akg.lang.cce.cast_to(res, dtype)
    return res


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def cosh(data):
    """
    cosh op for input tensor.

    ..math:`y = (e^(x)+e^(-x))/2`

    Args:
        data (tvm.tensor.Tensor): tensor with type float16 or float32.

    Returns:
        tvm.tensor.Tensor.
    """
    dtype = data.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    vc_util.check_shape(data.shape)

    return cosh_call(data)

