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

"""operator dsl function: rint"""

import akg.lang.cce
from akg.utils import validation_check as vc_util


def rint_compute(input_x):
    """rint compute implementation"""
    res = akg.lang.cce.round(input_x)
    res = akg.lang.cce.cast_to(res, input_x.dtype)

    return res


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def rint(input_x):
    """
    Calculating rint(x). returns the integer nearest to x by element-wise.
    If the result is between two representable values, the even number should be used.

    Args:
        input_x (tvm.tensor.Tensor): Tensor of float32, float16.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as input_x.
    """
    shape_x = input_x.shape
    dtype_x = input_x.dtype

    vc_util.check_shape(shape_x)
    vc_util.ops_dtype_check(dtype_x, vc_util.DtypeForDavinci.ALL_FLOAT)
    res = rint_compute(input_x)
    return res
