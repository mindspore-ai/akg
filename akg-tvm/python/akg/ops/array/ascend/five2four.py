#!/usr/bin/env python3
# coding: utf-8
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

"""operator dsl function: five2four"""
import akg.topi
from akg.tvm.hybrid import script
from akg.utils import custom_tiling as ct_util
import akg.utils as  utils
from akg.utils.format_transform import get_shape, get_bytes, to_tvm_const
from akg.utils.dynamic_shape import shape_is_dynamic


C_LIMIT_FOR_CAST = 3600


def get_attrs():
    """get attrs."""
    attrs = {
        "pragma_sink_last_axis": False
    }
    return attrs


def five2four_tiling_strategy(tensor, c_value=None, expansion=None):
    """Custom tiling strategy for five2four op."""
    strategy = list()
    if c_value is None:
        strategy = ct_util.create_template(tensor=tensor,
                                           template=ct_util.TileTemplate.NC1HWC0)
    elif not shape_is_dynamic(tensor):
        c_value = 16 if c_value < 16 else c_value
        node_n = ct_util.create_constraint_on_tensor(tensor=tensor,
                                                     values=1,
                                                     constraints=ct_util.TileConstraint.FACTOR,
                                                     tensor_pos=0)
        node_c1 = ct_util.create_constraint_on_tensor(tensor=tensor,
                                                      values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      tensor_pos=1)
        node_c0 = ct_util.create_constraint_on_tensor(tensor=tensor,
                                                      values=c_value,
                                                      constraints=ct_util.TileConstraint.FACTOR,
                                                      tensor_pos=4)
        strategy = node_n + node_c1 + node_c0
    if expansion:
        strategy.append(ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values=expansion,
                                                            constraints=ct_util.TileConstraint.SET_EXPANSION)[0])
    if shape_is_dynamic(tensor):
        # axis should be full tiled due to cast operator
        strategy.append(ct_util.modify_common_constraints(
                        value=0.85, constraint=ct_util.TileConstraint.SET_MEM_RATIO))
    return strategy


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), str, str, (str, type(None)))
def Five2Four(data, shape4d, dst_type, format_, target=utils.CCE):
    """
    Convert 5-dims "data" to 4-dims,the format of "data" is defined in "format_"

    Args:
        data (tvm.tensor.Tensor): 5-dims tensor of type float16, float32
        shape4d (Union[list, tuple]): a list has 4 nums, shape of output Tensor
        dst_type (str): data type of output Tensor
        format_ (str): a str defined the format of returns, support NCHW and NHWC

    Returns:
        4-dims tvm.tensor.Tensor.

    Supported Platforms:
        'Ascend'
    """
    utils.ops_dtype_check([data.dtype, dst_type], utils.DtypeForDavinci.ALL_FLOAT)
    shape5d = get_shape(data)
    if not shape_is_dynamic(data):
        if len(shape5d) != 5 or shape5d[-1] != 16:
            raise ValueError("five2four_cce only support 5-dim data and last dim should be 16")

    bs, c1, h, w, c0 = shape5d
    if not shape_is_dynamic(data):
        utils.davinci_format_check(shape5d, "NC1HWC0", dim=5)
    # Check format
    if format_ not in ['NCHW', 'NHWC']:
        raise ValueError("{} format is not support, five2four only support NCHW and NHWC format input"
                         .format(format_))
    if format_ == "NCHW":
        if shape_is_dynamic(data):
            shape4d = [bs, c1 * c0, h, w]
        _, c, h_4d, w_4d = shape4d
    else:
        if shape_is_dynamic(data):
            shape4d = [bs, h, w, c1 * c0]
        _, h_4d, w_4d, c = shape4d
    utils.davinci_format_check(shape4d, format_, dim=4)

    # Check is shape4d and shape5d match
    if False not in [isinstance(s, (int, akg.tvm.expr.IntImm)) for s in shape5d]:
        if h_4d != h or w_4d != w:
            raise ValueError("five2four_cce's shape4d h and w should equal to data shape's h and w")
        if c > c1 * c0 or c <= (c1 - 1) * c0:
            raise ValueError("five2four_cce's shape4d c should in set ((c1 - 1) * c0, c1 * c0]")

    # Check size c when casting happens
    if not shape_is_dynamic(data):
        if data.dtype != dst_type and c >= C_LIMIT_FOR_CAST:
            raise ValueError("When input and output data type is not matched, shape of 'c' axis should not exceed {}, "
                             "while currently set is {}".format(C_LIMIT_FOR_CAST, c))

    @script(capture=locals())
    def nc1hwc0_to_nhwc(inputs, bs, h, w, c, c1, c0):
        output = allocate((bs, h, w, c), inputs.dtype, "local")
        for n_i in range(bs):
            for h_i in range(h):
                for w_i in range(w):
                    for c_i in range(c1):
                        for c_i0 in range(c0):
                            output[n_i, h_i, w_i, c_i * c0 + c_i0] = inputs[n_i, c_i, h_i, w_i, c_i0]
        return output

    @script(capture=locals())
    def nc1hwc0_to_nchw(inputs, bs, h, w, c, c1, c0):
        output = allocate((bs, c, h, w), inputs.dtype, "local")
        for n_i in range(bs):
            for c_i in range(c1):
                for h_i in range(h):
                    for w_i in range(w):
                        for c_i0 in range(c0):
                            output[n_i, c_i * c0 + c_i0, h_i, w_i] = inputs[n_i, c_i, h_i, w_i, c_i0]
        return output

    # if c % 16 == 0, h and w == 1, five2four is a reshape operation
    if shape_is_dynamic(data):
        call_reshape = isinstance(h, int) and isinstance(w, int) and h == 1 and w == 1
    else:
        call_reshape = h == 1 and w == 1 and c % 16 == 0
    c_value = None
    expansion = None
    if format_ == "NHWC":
        if call_reshape:
            output = akg.topi.reshape(data, (bs, h, w, c))
            if shape_is_dynamic(data):
                output = akg.tvm.compute((bs, h, w, c), lambda *indice: output(*indice), name="reshape")
        elif c < c0:
            reshape_output = akg.topi.reshape(data, (bs, h, w, c0))
            output = akg.tvm.compute((bs, h, w, c), lambda *i: reshape_output(*i), name='slice_output')
        else:
            output = nc1hwc0_to_nhwc(
                data,
                to_tvm_const(bs),
                to_tvm_const(h),
                to_tvm_const(w),
                to_tvm_const(c),
                to_tvm_const(c1),
                to_tvm_const(c0))

    else:
        if call_reshape:
            output = akg.topi.reshape(data, (bs, c, h, w))
            if shape_is_dynamic(data):
                output = akg.tvm.compute((bs, c, h, w), lambda *indice: output(*indice), name="reshape")
        else:
            output = nc1hwc0_to_nchw(
                data,
                to_tvm_const(bs),
                to_tvm_const(h),
                to_tvm_const(w),
                to_tvm_const(c),
                to_tvm_const(c1),
                to_tvm_const(c0))

    # two special cases for tiling strategy
    if not shape_is_dynamic(data):
        if c < c0 or output.dtype != dst_type:
            c_value = c
        if c % c0 != 0 and output.dtype != dst_type:
            expansion = int(ct_util.BLOCK_SIZE / get_bytes(data.dtype))
    attrs = get_attrs()
    if not call_reshape:
        attrs["custom_tiling"] = five2four_tiling_strategy(data, c_value, expansion)

    if output.dtype != dst_type:
        output = akg.topi.cast(output, dst_type)
    return output, attrs
