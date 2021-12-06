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

"""operator dsl function:elu"""
import akg.topi
from akg import tvm
import akg.utils as utils
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.utils.kernel_exec import product_is_mini

def _elu_taylor_compute(data):
    """
    Calculate e^x - 1, Use fifth order taylor expansion

    e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!)
    e^x - 1 = x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!)

    Args:
        data (tvm.tensor.Tensor):  input

    Returns : 
        tvm.tensor.Tensor
    """
    TAYLOR_SECOND_ORDER_PARAM = 1 / 2.0
    TAYLOR_THIRD_ORDER_PARAM = 1 / 6.0
    TAYLOR_FOURTH_ORDER_PARAM = 1 / 24.0
    TAYLOR_FIFTH_ORDER_PARAM = 1 / 120.0

    dtype = data.dtype
    if dtype == "float16":
        data = akg.lang.ascend.cast_to(data, "float32")    

    # x^2 / 2!
    taylor_second_order_param = tvm.const(TAYLOR_SECOND_ORDER_PARAM, "float32")
    data_power_2 = akg.lang.ascend.vmul(data, data)
    data_power_2_div_2 = akg.lang.ascend.vmuls(data_power_2, taylor_second_order_param)

    # x^3 / 3!
    taylor_third_order_param = tvm.const(TAYLOR_THIRD_ORDER_PARAM, "float32")
    data_power_3 = akg.lang.ascend.vmul(data_power_2, data)
    data_power_3_div_6 = akg.lang.ascend.vmuls(data_power_3, taylor_third_order_param)

    # x^4 / 4!
    taylor_fourth_order_param = tvm.const(TAYLOR_FOURTH_ORDER_PARAM, "float32")
    data_power_4 = akg.lang.ascend.vmul(data_power_3, data)
    data_power_4_div_24 = akg.lang.ascend.vmuls(data_power_4, taylor_fourth_order_param)

    # x^5 / 5!
    taylor_fifth_order_param = tvm.const(TAYLOR_FIFTH_ORDER_PARAM, "float32")
    data_power_5 = akg.lang.ascend.vmul(data_power_4, data)
    data_power_5_div_120 = akg.lang.ascend.vmuls(data_power_5, taylor_fifth_order_param)

    res = akg.lang.ascend.vadd(data, data_power_2_div_2)
    res = akg.lang.ascend.vadd(res, data_power_3_div_6)
    res = akg.lang.ascend.vadd(res, data_power_4_div_24)
    res = akg.lang.ascend.vadd(res, data_power_5_div_120)

    if dtype == "float16":
        res = akg.lang.ascend.cast_to(res, "float16")
    return res

def _elu_mini_compute(exp_res, data, shape):
    """
    do element-wise e^x - 1 compute in mini scene

    f(x) = e^x - 1,                   x <= TAYLOR_THRESHOLD or x >= 0
    f(x) = fifth taylor computer,     TAYLOR_THRESHOLD < x < 0

    Args:
        exp_res (tvm.tensor.Tensor): the tensor of e^x -1, float16
        data (tvm.tensor.Tensor): input, float16
        shape (list): the shape of input

    Returns: 
        tvm.tensor.Tensor
    """
    TAYLOR_THRESHOLD = -0.7
    input_right_border = tvm.const(0.0, "float16")
    right_border = tvm.compute(shape, lambda *i: input_right_border)
    
    taylor_res = _elu_taylor_compute(data)

    input_left_border = tvm.const(TAYLOR_THRESHOLD, "float16")
    left_border = tvm.compute(shape, lambda *i: input_left_border)
    exp_taylor_neg = tvm.compute(shape, lambda *i: tvm.expr.Select\
                    (data(*i) > left_border(*i), taylor_res(*i), exp_res(*i)), name="gt")
    exp_res = tvm.compute(shape, lambda *i: tvm.expr.Select\
              (data(*i) < right_border(*i), exp_taylor_neg(*i), exp_res(*i)), name="lt")
    return exp_res

@utils.check_input_type(akg.tvm.tensor.Tensor)
def elu(data):
    """
    do element-wise elu operation

    f(x) = max(min(e^x - 1, 0), x), in cloud scene, for all inputs
    f(x) = max(min(e^x - 1, 0), x), in mini scene, for x <= TAYLOR_THRESHOLD or x >= 0
    f(x) = fifth taylor computer, in mini scene, for TAYLOR_THRESHOLD < x < 0

    Args:
        data (tvm.tensor.Tensor): tensor with type float16 or float32.

    Returns:
        tvm.tensor.Tensor.
    """
    dtype = data.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(data.shape)

    compute_dtype = dtype
    if dtype == "float16" and not product_is_mini():
        data = akg.lang.ascend.cast_to(data, "float32")
        compute_dtype = "float32"
    
    if dtype == "float32" and product_is_mini():
        data = akg.lang.ascend.cast_to(data, "float16")
        compute_dtype = "float16"

    input_border = tvm.const(0.0, compute_dtype)
    shape = get_shape(data.shape)
    tensor_input_border = tvm.compute(shape, lambda *i: input_border)

    exp_res = akg.topi.exp(data)
    exp_res = akg.lang.ascend.vadds(exp_res, -1)

    if product_is_mini():
        exp_res = _elu_mini_compute(exp_res, data, shape)

    negative_part = akg.lang.ascend.vmin(exp_res, tensor_input_border)
    res = akg.lang.ascend.vmax(negative_part, data)

    if dtype == "float16" and not product_is_mini():
        res = akg.lang.ascend.cast_to(res, "float16")
    if dtype == "float32" and product_is_mini():
        res = akg.lang.ascend.cast_to(res, "float32")
    return res
