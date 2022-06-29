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

"""operator dsl function:laplacian_of_gaussian"""
import akg.topi
import akg.tvm
import akg
from akg.ops.math import exp

def gaussian(x, sig=1.0, mean=0.0):
    r"""
    Implementation of gaussian filter.

    :math:`G(x,var) = 1/(2*pi*val^2) * exp(-\sum_j(x_{ij}^2)/(2*var^2))`
    """
    if (len(x.shape) == 1):
        two = akg.tvm.const(2, x.dtype)
        sig_cast = akg.tvm.const(sig, x.dtype)
        return 1 / (sig_cast * sig_cast * (akg.tvm.const(6.283, x.dtype))) * exp(-(x) * (x) / (two * sig_cast * sig_cast), target='cce')
    elif (len(x.shape) == 2):
        sig_cast = akg.tvm.const(sig, x.dtype)
        x_square = akg.tvm.compute(x.shape, lambda *i:  x(*i) * x(*i))
        sum_reduce = akg.topi.sum(akg.tvm.compute(x.shape, lambda *i: x_square(*i)*akg.tvm.const(-0.5, x.dtype)), axis=(1), keepdims=True)
        return 1 / (sig_cast * sig_cast * (akg.tvm.const(6.283, x.dtype))) * exp(sum_reduce, target='cce')
    else:
        raise RuntimeError("Do not support {0} dim laplacian of gaussian.".format(len(x.shape)))

def laplacian_of_gaussian_ad(head, x, target="cce"):
    """2nd derivative of gaussian, which should be the same as laplacian of gaussian filter."""
    y = gaussian(x)
    # 1st derivative
    dx = list(akg.differentiate(y, [x], head))
    head_fake = akg.tvm.compute(x.shape, lambda * ind: akg.tvm.const(1.0, dtype=y.dtype))
    # 2nd derivative
    dx2 = list(akg.differentiate(dx[0], [x], head_fake))
    return dx2[0]
