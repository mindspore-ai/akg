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

"""operator dsl function: atan2"""
import akg
import akg.utils as utils
from akg import tvm, topi
from akg.utils.format_transform import get_shape
from akg.utils.kernel_exec import product_is_mini
from akg.utils import dsl_create as dc
from .atan import atan
from ..reciprocal import reciprocal


def _init_atan2_mask(data_y_, data_x_):
    """
    Compute mask for atan2.

    Args:
        data_y (tvm.tensor.Tensor): The y of atan2(y, x).
        data_x (tvm.tensor.Tensor): The x of atan2(y, x).

    Returns:
        mask (tvm.tensor.Tensor): The mask of x's and y's value.
    """
    is_cast_for_mini = product_is_mini() and data_y_.dtype == "float32"

    # in mini, select only support float16
    if is_cast_for_mini:
        data_x = topi.cast(data_x_, "float16")
        data_y = topi.cast(data_y_, "float16")
    else:
        data_x = data_x_
        data_y = data_y_

    dtype_input = data_y.dtype

    tensor_one = dc.one_const(dtype_input)
    tensor_zero = dc.zero_const(dtype_input)
    tensor_neg_one = dc.neg_one_const(dtype_input)

    y_ge_zero = tvm.compute(data_y.shape,
                            lambda *i:
                            tvm.expr.Select(
                                data_y(*i) >= tensor_zero,
                                tensor_one, tensor_neg_one),
                            name="y_ge_zero")

    x_lt_zero_y_mask = tvm.compute(data_y.shape,
                                   lambda *i:
                                   tvm.expr.Select(
                                       data_x(*i) < tensor_zero,
                                       y_ge_zero(*i), tensor_zero),
                                   name="xlt0_y_mask")

    if is_cast_for_mini:
        x_lt_zero_y_mask = topi.cast(x_lt_zero_y_mask, "float32")
        y_ge_zero = topi.cast(y_ge_zero, "float32")

    return (x_lt_zero_y_mask, y_ge_zero)


def _atan2_compute(y, x):
    """compute for atan2"""
    const_pi_by_two = 1.5707963267948966192313216916398
    dtype = y.dtype
    if dtype == "float16":
        y = topi.cast(y, "float32")
        x = topi.cast(x, "float32")

    x_lt_zero_y_mask, y_ge_zero_mask = _init_atan2_mask(y, x)
    y_cmp_zero = topi.multiply(y_ge_zero_mask, tvm.const(const_pi_by_two, "float32"))
    res_x_lt_zero = topi.multiply(x_lt_zero_y_mask, dc.pi_const("float32"))

    # caculate the atan(y/x) when x > 0
    if product_is_mini():
        x_rec = reciprocal(x, target=utils.CCE)
        res = topi.multiply(y, x_rec)
    else:
        res = topi.divide(y, x)
    res, _ = atan(res)

    if product_is_mini():
        tensor_zero = dc.zero_const("float16")
        x = topi.cast(x, "float16")
        y_cmp_zero = topi.cast(y_cmp_zero, "float16")
        res = topi.cast(res, "float16")
    else:
        tensor_zero = dc.zero_const("float32")

    res = tvm.compute(res.shape,
                      lambda *i:
                      tvm.expr.Select(x(*i) == tensor_zero,
                                      y_cmp_zero(*i), res(*i)),
                      name="res")

    if product_is_mini():
        res = topi.cast(res, "float32")

    res = topi.add(res, res_x_lt_zero)
    return topi.cast(res, dtype)


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def atan2(y, x):
    """
    Compute arc tangent of y/x.

    .. math::
        \\arctan2(y, x) = \\arctan(\\frac{y}{x})

    Args:
        y (tvm.tensor.Tensor): Input tensor, only support float16, float32.
        x (tvm.tensor.Tensor): Input tensor, only support float16, float32.

    Returns:
        A tvm.tensor.Tensor as angles in radians.

    Supported Platforms:
        'Ascend'
    """
    utils.elemwise_shape_check(get_shape(y), get_shape(x))
    utils.elemwise_dtype_check(y.dtype, x.dtype, utils.DtypeForDavinci.ALL_FLOAT)

    return _atan2_compute(y, x), {"enable_auto_inline": False}
