# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: atan"""
import akg
import akg.utils as utils
import akg.utils.dsl_create as dc
from akg import tvm, topi
from akg.utils.format_transform import get_shape
from akg.utils.kernel_exec import product_is_mini

CONST_PI_BY_FOUR = 0.78539816339744830961566084581988
CONST_PI_BY_EIGHT = 0.39269908169872415480783042290994
CONST_ITERTOR = 6
CONST_ITERTOR2 = 4
TAN_PI_BY_EIGHT = 0.4142135623730950

ATAN_TAYLOR_COEF = (1.0, -1.0 / 3, 1.0 / 5, -1.0 / 7, 1.0 / 9, -1.0 / 11, 1.0 / 13)


def _do_atan_taylor(data):
    """
    Taylor algorithm for atan.

        if x > 0 and x < tan(pi/8):
            atan(x) = x - x^3/3 + x^5/5 - x^7/7 ...
        elif x > tan(pi/8) and x < tan(pi/4):
            atan(x) = atan(y) + atan((x-y)/(1+xy))

    Args:
        data (tvm.tensor.Tensor): Input data.

    Returns:
        A tvm.tensor.Tensor of atan(x).
    """
    dtype = data.dtype

    tensor_offset = tvm.const(TAN_PI_BY_EIGHT, dtype)
    deno = topi.multiply(data, tvm.const(TAN_PI_BY_EIGHT, dtype))
    deno = topi.add(deno, dc.one_const(dtype))
    molecule = topi.subtract(data, tensor_offset)
    ddata = topi.divide(molecule, deno)
    ddata = topi.abs(ddata)

    square_ddata = topi.multiply(ddata, ddata)
    res = tvm.const(ATAN_TAYLOR_COEF[CONST_ITERTOR], dtype)
    for i in reversed(range(CONST_ITERTOR)):
        res = topi.multiply(res, square_ddata)
        res = topi.add(res, tvm.const(ATAN_TAYLOR_COEF[i], dtype))
    res = topi.multiply(res, ddata)
    res = topi.add(res, tvm.const(CONST_PI_BY_EIGHT, dtype))

    square_data = topi.multiply(data, data)
    res2 = tvm.const(ATAN_TAYLOR_COEF[CONST_ITERTOR2], dtype)
    for i in reversed(range(CONST_ITERTOR2)):
        res2 = topi.multiply(res2, square_data)
        res2 = topi.add(res2, tvm.const(ATAN_TAYLOR_COEF[i], dtype))
    return topi.minimum(res, topi.multiply(res2, data))


def _atan_compute(data):
    """compute for atan"""
    dtype = data.dtype

    if dtype == "float16":
        data = topi.cast(data, "float32")

    abs_data = topi.abs(data)
    tensor_one = dc.one_const(abs_data.dtype)

    abs_data_sub_one = topi.subtract(abs_data, tensor_one)
    abs_data_add_one = topi.add(abs_data, tensor_one)
    abs_data2 = topi.abs(topi.divide(abs_data_sub_one, abs_data_add_one))

    # calucate data less than one
    res = _do_atan_taylor(abs_data)
    # calucate data more than one
    res_mt_one = topi.add(_do_atan_taylor(abs_data2),
                          tvm.const(CONST_PI_BY_FOUR, abs_data2.dtype))
    res = topi.minimum(res, res_mt_one)

    if product_is_mini() and data.dtype == "float32":
        sign_mask = topi.cast(
            topi.sign(topi.cast(data, "float16")), "float32")
    else:
        sign_mask = topi.sign(data)

    res = topi.multiply(res, sign_mask)

    if dtype == "float16":
        res = topi.cast(res, "float16")

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def atan(x):
    """
    Compute trigonometric inverse tangent of x.

    Args:
        x (tvm.tensor.Tensor): Input tensor, only support float16, float32.

    Returns:
        A tvm.tensor.Tensor as result of atan.
    
    Supported Platforms:
        'Ascend'
    """
    utils.check_shape(get_shape(x))
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)

    return _atan_compute(x), {"enable_auto_inline": False}
