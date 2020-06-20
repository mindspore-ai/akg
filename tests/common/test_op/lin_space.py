# Copyright 2020 Huawei Technologies Co., Ltd
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
import akg.lang.cce
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.ops.math.div import div

def lin_space_compute(input_assist, input_start, input_stop, input_num):
    """inv_grad compute implementation"""
    num_float = akg.lang.cce.cast_to(input_num, "float32")
    num_divided = akg.lang.cce.vadds(num_float, -1.0)

    step_divider = akg.lang.cce.vsub(input_stop, input_start)
    step = div(step_divider, num_divided)

    res_temp = akg.lang.cce.vmul(input_assist, akg.lang.cce.broadcast(step, input_assist.shape))
    res = akg.lang.cce.vadd(res_temp, akg.lang.cce.broadcast(input_start, input_assist.shape))

    return res



@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
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
    vc_util.check_shape(shape_assist)

    # check the data type, start type only support float32,num type only support int32
    vc_util.ops_dtype_check(dtype_input, vc_util.DtypeForDavinci.FLOAT32)
    vc_util.ops_dtype_check(dtype_num, vc_util.DtypeForDavinci.INT32)

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

