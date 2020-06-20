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

"""operator dsl function:blas_axby"""

import akg
from akg.utils import validation_check as vc_util

@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (float, int), (float, int))
def blas_axby(x, y, alpha, beta):
    r"""
    Blas axby.

    :math:`\alpha x + \beta y`

    Args:
        x (tvm.tensor.Tensor): Input `x` of type float16 or float32.
        y (tvm.tensor.Tensor): Input `y` of type float16 or float32.
        alpha (Union[int, float]): Scale of `x`.
        beta (Union[int, float]): Scale of `y`.

    Returns:
        tvm.tensor.Tensor, has the same shape and type as inputs.
    """
    vc_util.ops_dtype_check([x.dtype, y.dtype], vc_util.DtypeForDavinci.ALL_FLOAT)
    vc_util.check_shape(x.shape)
    vc_util.check_shape(y.shape)

    ax = akg.lang.cce.vmuls(x, alpha)
    by = akg.lang.cce.vmuls(y, beta)
    res = akg.lang.cce.vadd(ax, by)

    return res
