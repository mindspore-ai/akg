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

"""operator dsl function:prelu"""

import akg.tvm
import akg.topi
from akg.utils import validation_check as vc_util
from akg.lang.cce import vmax, vmin, vadd, vmul, broadcast


def prelu(A, w):
    """
    brief Computes prelu value of a tensor.

    \f[
    out = max(0, A) + min(0, wA)
    \f]
    Args:
         inputs akg.tvm.Tensor of type float16, float32

    Returns:
         akg.tvm.Tensor of same type and shape as inputs
    """
    # num_parameters=1, init=0.25
    shape1 = [x.value for x in A.shape]
    dtype1 = A.dtype
    shape2 = [x.value for x in w.shape]
    dtype2 = w.dtype
    assert len(shape1) == 4, "only support 4-dim pooling"  # NCHW
    assert len(shape2) == 1, "only support 1-dim a"
    assert (shape2[0] == shape1[1] or shape2[0] == 1), "there is only two values are legitimate: 1, or the number of channels at input. Default: 1"

    check_list = ["float16", "float32"]
    if not (dtype1.lower() in check_list and dtype2.lower() in check_list):
        raise RuntimeError("tile_cce only support %s while dtype is %s and %s" % (",".join(check_list), dtype1, dtype2))
    vc_util.check_shape(shape1)
    vc_util.check_shape(shape2)

    w_reshape = akg.topi.reshape(w, (1, shape2[0], 1, 1))
    w_broadcast = akg.topi.broadcast_to(w_reshape, shape1)

    const_zero = broadcast(akg.tvm.const(0, A.dtype), shape1, output_dtype=A.dtype)

    res = vadd(vmax(A, const_zero), vmul(vmin(A, const_zero), w_broadcast))

    return res
