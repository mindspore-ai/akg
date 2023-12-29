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

"""operator dsl function: lin_space"""
import akg.lang.ascend
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.ops.math import divide

def lin_space_compute(input_assist, input_start, input_stop, input_num):
    """inv_grad compute implementation"""
    num_float = akg.lang.ascend.cast_to(input_num, "float32")
    num_divided = akg.lang.ascend.vadds(num_float, -1.0)

    step_divider = akg.lang.ascend.vsub(input_stop, input_start)
    step = divide(step_divider, num_divided, target="cce")

    res_temp = akg.lang.ascend.vmul(input_assist, akg.lang.ascend.broadcast(step, input_assist.shape))
    res = akg.lang.ascend.vadd(res_temp, akg.lang.ascend.broadcast(input_start, input_assist.shape))

    return res



@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def lin_space(input_assist, input_start, input_stop, input_num):
    """
    Generates values in an interval.
    A sequence of 'num' evenly-spaced values are generated beginning at 'start'.
    If 'num' > 1, the values in the sequence increase by 'stop - start / num - 1',
    so that the last one is exactly 'stop'.

    Note:
        input_assist shape must be 1D, and the number equal to input_num.

    Args:
        input_assist (tvm.tensor.Tensor): Tensor of float32.
        input_start (tvm.tensor.Tensor): Tensor of float32.
        input_stop (tvm.tensor.Tensor): Tensor of float32.
        input_num (tvm.tensor.Tensor): Tensor of int32.

    Retruns:
        tvm.tensor.Tensor, has the same type and shape as input_assist.

    """

    shape_assist = input_assist.shape
    shape_start = input_start.shape
    shape_stop = input_stop.shape
    shape_num = input_num.shape
    dtype_input = input_start.dtype
    dtype_num = input_num.dtype

    # check shape
    utils.check_shape(shape_assist)

    # check the data type, start type only support float32,num type only support int32
    utils.ops_dtype_check(dtype_input, utils.DtypeForDavinci.FLOAT32)
    utils.ops_dtype_check(dtype_num, utils.DtypeForDavinci.INT32)

    # check shape of start, stop and num, must be (1,)
    shape_start = tuple(get_shape(shape_start))
    shape_stop = tuple(get_shape(shape_stop))
    shape_num = tuple(get_shape(shape_num))
    if shape_start != (1,) or shape_stop != (1,) or shape_num != (1,):
        raise ValueError(
            "lin_space only support rank=1 while shape of start or stop or num is not (1,)")

    # check shape of assist, only support 1dim
    if len(shape_assist) != 1:
        raise ValueError(
            "lin_space only support rank=1 while length of assist shape is %d"
            % (len(shape_assist)))

    res = lin_space_compute(input_assist, input_start, input_stop, input_num)
    return res

