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

"""operator dsl function: rsqrt_grad"""


import akg.tvm
from akg.utils import validation_check as vc_util


def rsqrt_grad(y, dy):
    """
    Computes the gradient reciprocal of square root of x element-wise.
    \f[
    dx = -0.5 * dy * x^{-\frac{3}{2}}
    \f]

    Note:
         In this function Y equal to 1/sqrt(x), so the formula can be define as:
             \f[ dx = -0.5 * dy * Y^3 \f]

    Args:
        Y (tvm.tensor.Tensor): Tensor of type float16, float32.
        dy (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has same type and shape as Y.
    """

    # Check type
    check_list = ["float16", "float32"]
    dtype = y.dtype
    if not dtype in check_list:
        raise RuntimeError("rsqrt_grad_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    dtype = dy.dtype
    if not dtype in check_list:
        raise RuntimeError("rsqrt_grad_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    # Check shape
    shape = [x.value for x in y.shape]
    vc_util.check_shape(shape)

    # dx = dy * (-0.5) * Y^3
    vpow2_t = akg.tvm.compute(shape, lambda *indice: y(*indice) * y(*indice), name="vpow2_t")
    vpow3_t = akg.tvm.compute(shape, lambda *indice: vpow2_t(*indice) * y(*indice), name="vpow3_t")
    aa = akg.tvm.const(-0.5, dtype=dtype)
    vmuls_t = akg.tvm.compute(shape, lambda *indice: vpow3_t(*indice) * aa, name="vmuls_t")
    dx = akg.tvm.compute(shape, lambda *indice: dy(*indice) * vmuls_t(*indice), name="dx")

    return dx
