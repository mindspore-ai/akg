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

"""operator dsl function:gelu_grad"""
import akg
import akg.lang.ascend
import akg.tvm
from akg.ops.math.ascend import Tanh as tanh_ATT


def gelu_grad(x, dy, target="cce"):
    check_list = ["float16", "float32"]
    dtype = x.dtype
    if not dtype.lower() in check_list:
        raise RuntimeError("gelu_grad_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = [tmp.value for tmp in x.shape]

    t = akg.lang.ascend.broadcast(akg.tvm.const(0.044715, dtype), shape)
    s = akg.lang.ascend.broadcast(akg.tvm.const(0.7978846, dtype), shape)  # sqrt(2/pi)
    one = akg.lang.ascend.broadcast(akg.tvm.const(1, dtype), shape)
    half = akg.lang.ascend.broadcast(akg.tvm.const(0.5, dtype), shape)
    three = akg.lang.ascend.broadcast(akg.tvm.const(3, dtype), shape)

    x_pow2 = akg.lang.ascend.vmul(x, x)
    x_pow3 = akg.lang.ascend.vmul(x_pow2, x)

    # tanh = np.tanh(s * (x + t * x * x * x))
    tanh_data = akg.tvm.compute(shape, lambda *i: s(*i) * (x(*i) + t(*i) * x_pow3(*i)), name="tanh_data")
    tanh = tanh_ATT(tanh_data)
    tanh_pow2 = akg.lang.ascend.vmul(tanh, tanh)

    # dy/dx = 0.5 * (1 + tanh + x * (1 - tanh*tanh) * s * (1 + 3*t*x*x))
    grad = akg.tvm.compute(shape, lambda *i: half(*i) * (one(*i) + tanh(*i) +
                                                     x(*i) * (one(*i) - tanh_pow2(*i)) * s(*i) *
                                                     (one(*i) + three(*i) * t(*i) * x_pow2(*i))), name="grad")

    dx = akg.lang.ascend.vmul(grad, dy)
    return dx
