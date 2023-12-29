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

"""operator dsl function: gelu_ad"""
import akg.tvm
import akg.topi
import akg
from akg.ops.math.ascend import Tanh
from tests.common.test_op.ascend import gelu

def tanh_fdiff(head, inp):
    """
    In order to achieve higher precision, we self-define differentiate with simplify calculation.

    .. math::

    \\frac{d_{tanh}}{d_x} = 4e^{-2x} / (1+2e^{-2x}+e^{-4x})
    """
    data_abs = akg.topi.abs(inp)
    dtype = inp.dtype
    exp_2 = data_abs * akg.tvm.const(-2.0, dtype)
    exp_4 = data_abs * akg.tvm.const(-4.0, dtype)
    exp_2_value = akg.topi.exp(exp_2)
    exp_4_value = akg.topi.exp(exp_4)
    exp_2_value_2 = exp_2_value * akg.tvm.const(2.0, dtype)
    exp_2_value_4 = exp_2_value * akg.tvm.const(4.0, dtype)
    sum_dino_exp = akg.tvm.const(1.0, dtype) + exp_2_value_2 + exp_4_value
    dep_tanh = exp_2_value_4/sum_dino_exp
    res = akg.topi.multiply(head, dep_tanh)
    return res

def gelu_ad(head, in_data, target="cce"):
    """Compute gradient for gelu operator using automatic differentiate."""
    res = gelu.gelu(in_data)
    jacs = list(akg.differentiate(res, [in_data], head))
    return jacs[0]

def gelu_ad_custom(head, in_data, target="cce"):
    """
    Automatic differentiation of gelu with customize function.

    In order to achieve higher precision, we could also self-define tanh part differentiate with simplify calculation.
    """
    dtype = in_data.dtype
    const1 = akg.tvm.const(0.044715, dtype)
    const2 = akg.tvm.const(0.7978845, dtype)
    const3 = akg.tvm.const(0.1070322, dtype)
    tmp0 = akg.topi.multiply(in_data, in_data)
    pow0 = akg.topi.multiply(tmp0, in_data)
    mul0 = pow0 * const1
    add0 = in_data + mul0
    mul1 = add0 * const2
    tanh_res = Tanh(mul1)
    add1 = tanh_res + akg.tvm.const(1, dtype)
    mul2 = add1 * akg.tvm.const(0.5, dtype)
    mul3 = in_data * mul2
    res = mul3

    def gelu_diff(out, inp, head, ad_attrs, new_array_pld):
        temp = tanh_fdiff(head, mul1)
        return [temp * (akg.tvm.const(0.7978845, dtype) + const3*inp[0]*inp[0])]
    jacs = list(akg.differentiate(res, [in_data], head, None, None, override={tanh_res: ([in_data], gelu_diff)}))
    return jacs[0]

