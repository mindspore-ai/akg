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

"""operator dsl function: truncate_div"""
import akg
from akg import tvm, topi
from akg.utils.format_transform import get_shape
import akg.utils as utils
from akg.utils import dsl_create as dc
from akg.ops.math import Cast
from akg.ops.math.ascend import floor, ceil
from akg.utils.kernel_exec import product_is_mini


def truncate_div_compute(input_x1, input_x2):
    """compute for truncate_div"""
    int_list = ("int32", "int8", "uint8")

    if input_x1.dtype in int_list:
        data_zero = dc.zero_const("float32")
        data_x_broad = Cast(input_x1, "float32", target=utils.CCE)
        data_y_broad = Cast(input_x2, "float32", target=utils.CCE)
        res_div = topi.divide(data_x_broad, data_y_broad)
        res_min_int = ceil(topi.minimum(res_div, data_zero))
        res_max_int = floor(topi.maximum(res_div, data_zero))
        res_trunc = topi.add(res_min_int, res_max_int)
        res_trunc = Cast(res_trunc, "float32", target=utils.CCE)
    else:
        res_trunc = topi.divide(input_x1, input_x2)

    return Cast(res_trunc, input_x1.dtype, target=utils.CCE)


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def truncate_div(input_x1, input_x2):
    """
    Calculating data's truncate_div, res = floor(x1/x2) if x1/x2>0 else ceil(x1/x2).

    Args:
        input_x1 (tvm.tensor.Tensor): Input tensor, support float16,
                                      float32 on mini device, while support
                                      int32, int8, uint8, float16, float32 on
                                      cloud ones.
        input_x2 (tvm.tensor.Tensor): Input tensor, with same dtype as input_x1.
    Returns:
        A tvm.tensor.Tensor as result of truncate_div.
    """
    utils.check_shape(get_shape(input_x1))
    utils.check_shape(get_shape(input_x2))
    utils.elemwise_dtype_check(input_x1.dtype, input_x2.dtype)
    utils.ops_dtype_check(
        input_x1.dtype,
        (utils.DtypeForDavinci.ALL_FLOAT) if product_is_mini() \
            else (utils.DtypeForDavinci.ALL_FLOAT,
                  utils.DtypeForDavinci.INT32,
                  utils.DtypeForDavinci.INT8,
                  utils.DtypeForDavinci.UINT8))

    return truncate_div_compute(input_x1, input_x2)
