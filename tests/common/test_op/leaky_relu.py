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

"""operator dsl function:leaky_relu"""
import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils

@vc_util.check_input_type(akg.tvm.tensor.Tensor, (int, float, None))
def leaky_relu(data, negative_slop=0):
    """
    leaky_relu op for input tensor (N,C,H,W) OR (N,C1,H,W,C0).

    ..math:`max(x,negative_slop*x)`

    Args:
        data (tvm.tensor.Tensor): tensor with type float16 or float32.
        negative_slop (float): 0<=negative_slop<1

    Returns:
        tvm.tensor.Tensor.
    """
    dtype = data.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    vc_util.check_shape(data.shape)

    if negative_slop >= 1 or negative_slop < 0:
        raise RuntimeError(
            "leaky_relu only support negative_slop between [0,1)")

    slop_tmp = akg.tvm.const(negative_slop, dtype=dtype)
    tmp = akg.lang.cce.vmuls(data, slop_tmp)
    res = akg.lang.cce.vmax(tmp, data)

    return res


