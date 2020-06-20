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

"""operator dsl function: cos"""
import akg
import akg.tvm
import akg.topi
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util


NEG_NUM_ONE = -1.0
pi = akg.tvm.const(3.14159265358979, dtype="float32")
tylor_list = [0.5000000000, 0.0416666666667, 0.0013888888889, 0.000024801587301]

def factorial(n):
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res

def cos_zero(input):
    """Computes cosine value of a tensor.

    :math: `cos(x) = 1-x^2(1/2-x^2(1/4!-x^2(1/6!-x^2(1/8!-1/10!*x^2(...)))))`

    Args:
        input (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor of same type and shape as in_data.
    """
    input_square = akg.tvm.compute(input.shape, lambda *i: input(*i) * input(*i), name="input_square")
    tylor_list_len = len(tylor_list)
    mid = akg.tvm.compute(input.shape,
                          lambda *index: input_square(*index) * tylor_list[-1],
                          name="mid_res_last")

    for i, tylor_value in reversed(list(enumerate(tylor_list[:-1]))):
        name = "mid_res" + str(tylor_list_len - 1 - i)
        mid = akg.tvm.compute(input.shape,
                              lambda *index: input_square(*index) * (tylor_value - mid(*index)),
                              name=name)
    res = akg.tvm.compute(input.shape, lambda *index: akg.tvm.const(1.0, dtype="float32") - mid(*index), name="res")

    return res

def get_attrs():
    """get attrs."""
    attrs = {
        "enable_feature_library": True
    }
    return attrs

def cos_dsl(input_x):

    """Compute cosine value of a tensor."""
    type_x = input_x.dtype
    check_list = ["float16", "float32"]
    if not (type_x.lower() in check_list):
        raise RuntimeError("cos only support %s while dtype is %s" % (",".join(check_list), type_x))
    vc_util.check_shape(input_x.shape)
    if type_x == "float16":
        input_x = akg.lang.cce.cast_to(input_x, "float32")


    pi_multiple = akg.lang.cce.vmuls(input_x, 1 / pi)
    round_float = akg.lang.cce.cast_to(akg.lang.cce.round(pi_multiple), "float32")
    # to adjust x to [-pi/2, pi/2]
    trans_x = akg.lang.cce.vsub(input_x, akg.lang.cce.vmuls(round_float, pi))
    res_trans_x = cos_zero(trans_x)
    res_mid = res_trans_x

    # if round is odd, the final result need to mutiply -1.
    # Need to multipy 1/2 to get the ceil value
    ceil_value = akg.lang.cce.ceil(akg.lang.cce.vmuls(round_float, 1 / 2))
    # if odd, ceil*2-round is 1,if even, the value is 0
    sub_value = akg.lang.cce.vsub(
        akg.lang.cce.vmuls(ceil_value, akg.tvm.const(2.0, "float32")), round_float)
    tensor_one = akg.lang.cce.broadcast(akg.tvm.const(1.0, "float32"), input_x.shape)
    odd_tensor = akg.lang.cce.vsub(tensor_one, sub_value)
    even_tensor = akg.lang.cce.vsub(odd_tensor, tensor_one)
    odd_even_tensor = akg.lang.cce.vadd(odd_tensor, even_tensor)
    res = akg.lang.cce.vmul(res_mid, odd_even_tensor)

    if type_x == "float16":
        res = akg.lang.cce.cast_to(res, "float16")
    return res

def cos(input_x):
    """Compute cosine value of a tensor."""
    dtype = input_x.dtype
    shape = get_shape(input_x)
    vc_util.ops_dtype_check(input_x.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    vc_util.check_shape(input_x.shape)
    if dtype == "float16":
        input_x = akg.lang.cce.cast_to(input_x, "float32")

    res = akg.tvm.compute(shape, lambda *indice: akg.lang.cce.cos(input_x(*indice)), name="res")

    # cast the dtype to float16
    if dtype == "float16":
        res = akg.lang.cce.cast_to(res, "float16")

    return res, get_attrs()