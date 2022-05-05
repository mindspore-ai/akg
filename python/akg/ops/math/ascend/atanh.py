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

"""operator dsl function: atanh"""
import akg
import akg.utils as utils
import akg.utils.dsl_create as dc
from akg import tvm, topi
from akg.utils.kernel_exec import product_is_mini
from akg.utils.format_transform import get_shape
from ..log import log


def _compute_taylor(data_input):
    """
    Algorithm: atanh(x) value is (x + x^3/3 +  x^5/5 +  x^7/7)
    Taylor 1/3 + x^2(1/5 + x^2/7)
    Taylor 1 + x^2(1/3 + x^2(1/5 + x^2/7))
    Taylor x(1 + x^2(1/3 + x^2(1/5 + x^2/7)))
    """

    taylor_para = [0, 1.0,  0, 1/3.0, 0, 1.0/5, 0, 1.0/7]
    # x^2
    data_mul_2 = topi.multiply(data_input, data_input)
    # 1/5 + x^2/7
    data_mul_2_7 = topi.multiply(data_mul_2, tvm.const(taylor_para[7], "float32"))
    result = topi.add(data_mul_2_7, tvm.const(taylor_para[5], "float32"))
    result = topi.multiply(data_mul_2, result)
    result = topi.add(result, tvm.const(taylor_para[3], "float32"))
    result = topi.multiply(data_mul_2, result)
    result = topi.add(result, tvm.const(taylor_para[1], "float32"))
    return topi.multiply(data_input, result)


def _compute_log(data_input, target=utils.CCE):
    """atanh(x) value is 0.5*log((1+x)/(1-x))"""

    data_1_sum_x = topi.add(data_input, dc.one_const(data_input.dtype))
    data_sub_x = topi.multiply(data_input, dc.neg_one_const(data_input.dtype))
    data_1_sub_x = topi.add(data_sub_x, dc.one_const(data_input.dtype))
    data_x_mul = data_1_sum_x / data_1_sub_x
    data_x_log = log(data_x_mul, target)
    data_res = topi.multiply(data_x_log, dc.half_const(data_input.dtype))

    return data_res


def _compute_mini(data_input, shape):
    """
    Use log and taylor to compute
    arctanh has the feature: arctanh(-abs(x)) = -arctanh(abs(x))
    """

    data_abs = topi.abs(data_input)
    result_ln = _compute_log(data_abs)
    result_taylor = _compute_taylor(data_abs)

    data_abs = topi.cast(data_abs, "float16")
    data_input = topi.cast(data_input, "float16")
    result_taylor = topi.cast(result_taylor, "float16")
    result_ln = topi.cast(result_ln, "float16")
    # when |x| < 0.5 using taylor computing, and when 0.5<|x|<1 using log()
    data_res = tvm.compute(shape,
                           lambda *i : akg.tvm.expr.Select(data_abs(*i) < dc.half_const("float16"),
                                                           result_taylor(*i),
                                                           result_ln(*i)),
                           name="le")

    # arctanh has the feature: arctanh(-abs(x)) = -arctanh(abs(x))
    data_res_neg = topi.multiply(data_res, dc.neg_one_const("float16"))
    data_res = tvm.compute(shape,
                           lambda *i : akg.tvm.expr.Select(data_input(*i) < dc.zero_const("float16"),
                                                           data_res_neg(*i),
                                                           data_res(*i)),
                           name="neg")
    return data_res


def _compute_cloud(data):
    """Use log to compute"""
    return _compute_log(data)


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def atanh(input_data):
    """
    Return atanh(x)=0.5*ln((1+x)/(1-x)) if abs(x)<1.

    Args:
        input_data (tvm.tensor.Tensor): Input tensor, only support float16, float32.

    Returns:
        A tvm.tensor.Tensor as result of atanh.

    Supported Platforms:
        'Ascend'
    """
    shape = get_shape(input_data)
    utils.check_shape(shape)

    inp_dtype = input_data.dtype
    utils.ops_dtype_check(inp_dtype, utils.DtypeForDavinci.ALL_FLOAT)

    if inp_dtype == "float16":
        input_data = topi.cast(input_data, "float32")

    if product_is_mini():
        res = _compute_mini(input_data, shape)
    else:
        res = _compute_cloud(input_data)

    res = topi.cast(res, inp_dtype)

    return res
