# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: add"""
import akg.topi
import akg.tvm
from akg.lang.ascend import vadd, vmuls
import akg.utils as utils
from akg.utils.dsl_create import produce_shapes
from akg.utils.format_transform import get_shape
from akg.utils.dynamic_shape import shape_is_dynamic


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def _add(data1, data2):
    utils.elemwise_dtype_check(data1.dtype, data2.dtype)
    utils.check_shape(data1.shape)
    utils.check_shape(data2.shape)
    utils.auto_broadcast_check(data1.shape, data2.shape)

    res = akg.topi.add(data1, data2)

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          (int, float, type(None)), (bool, type(None)), (dict, type(None)))
def _add_ascend(first_input, second_input, scale=1.0, polyhedral=True, attrs=None):
    """
    Computes first_input + second_input * scale elementwise.

    Args:
        first_input (tvm.tensor.Tensor): Tensor of type float16, float32, int32.
        second_input (tvm.tensor.Tensor): Tensor with same type as first_input.
                                      Broadcast will happen if shapes of input tensors are different.
        scale (float): scale factor applied on second_input, default value is 1.0.
        polyhedral (bool): If True, use auto-schedule, else use manual-schedule, default value is True.
        attrs (dict): Specifies parameters used in manual-schedule.

    Returns:
        tvm.tensor.Tensor of same type as input tensor with shape the broadcast shape of input tensors.
    """
    utils.check_shape(first_input.shape)
    utils.check_shape(second_input.shape)

    first_input_shape = get_shape(first_input)
    second_input_shape = get_shape(second_input)

    if shape_is_dynamic([first_input, second_input]):
        if first_input_shape != second_input_shape:
            raise RuntimeError("Input tensors have different shapes, broadcast is not supported for dynamic.")
        first_broadcast = first_input
        second_broadcast = second_input
    else:
        if first_input_shape != second_input_shape:
            _, _, out_shape = produce_shapes(first_input_shape, second_input_shape)
        else:
            out_shape = first_input_shape
        first_broadcast = akg.topi.broadcast_to(first_input, out_shape)
        second_broadcast = akg.topi.broadcast_to(second_input, out_shape)

    first_input_type = first_input.dtype
    second_input_type = second_input.dtype
    if first_input_type != second_input_type:
        raise TypeError("Input tensors have different data types.")
    utils.ops_dtype_check(first_input_type, utils.DtypeForDavinci.ALL_TYPES)

    temp = vmuls(second_broadcast, scale)
    res = vadd(first_broadcast, temp)
    res_cast = res.astype(first_input_type)
    if polyhedral:
        return res_cast

    def comp_func(s):
        first_ub = s.cache_read(first_input, "local.UB", [first_broadcast])
        second_ub = s.cache_read(second_input, "local.UB", [second_broadcast])
        res_cast_ub = s.cache_write(res_cast, "local.UB")

        s[first_broadcast].set_scope("local.UB")
        s[second_broadcast].set_scope("local.UB")
        s[temp].set_scope("local.UB")
        s[res].set_scope("local.UB")

        split_axis = []
        for i in range(len(attrs["tile"])):
            outer, inner = s[res_cast].split(res_cast.op.axis[i], attrs["tile"][i])
            axis_dict = {"outer": outer, "inner": inner}
            split_axis.append(axis_dict)

        s[first_ub].compute_at(s[res], res.op.axis[0])
        s[second_ub].compute_at(s[res], res.op.axis[0])

        s[first_broadcast].compute_at(s[res], res.op.axis[0])
        s[second_broadcast].compute_at(s[res], res.op.axis[0])

        s[temp].compute_at(s[res], res.op.axis[0])
        s[res].compute_at(s[res_cast_ub], res_cast_ub.op.axis[0])

        s[res_cast_ub].compute_at(s[res_cast], split_axis[-1]['outer'])

        # no scaling nedeed
        if scale == 1:
            s[temp].compute_inline()

        # no broadcast needed
        if first_input_shape == second_input_shape:
            s[first_broadcast].compute_inline()
            s[second_broadcast].compute_inline()

    return res_cast, comp_func


def add(data1, data2, scale=1.0, polyhedral=True, attrs=None, target=utils.CCE):
    """
    Computes data1 + data2 elementwise, broadcast is supported.

    Args:
        data1 (tvm.tensor.Tensor): Tensor.
        data2 (tvm.tensor.Tensor): Tensor of same type as data1, if shape(data2) != shape(data1), broadcast will happen.

    Returns:
        tvm.tensor.Tensor, add result, with same type as input tensors and broadcasted shape of data1 and data2.
    
    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return _add_ascend(data1, data2, scale, polyhedral, attrs)
    else:
        return _add(data1, data2)

