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

"""acos_grad"""

import akg.tvm
import akg.utils as utils
from ..rsqrt import rsqrt


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def acos_grad(x, dy, target=utils.CCE):
    """
    Gradient for acos.

    .. math:
        dx = [\\frac{-1}{(1 - x^2)^0.5} / ] \\cdot dy

    Args:
        x (tvm.tensor.Tensor): tensor of type float16, float32.
        dy (tvm.tensor.Tensor): tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, same type and shape as x.
    
    Supported Platforms:
        'Ascend'
    """
    dtype = x.dtype
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.ops_dtype_check(dy.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(x.shape)
    utils.check_shape(dy.shape)

    one = akg.tvm.const(1.0, dtype=dtype)
    mid_square = akg.tvm.compute(x.shape, lambda *i: (one - x(*i) * x(*i)), name="mid_square")
    rsq = rsqrt(mid_square, target)
    dx = akg.tvm.compute(x.shape, lambda *i: -rsq(*i) * dy(*i), name="dx")

    return dx
