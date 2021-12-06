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

"""operator dsl function: four2five"""
import akg
import akg.tvm
from akg.tvm.hybrid import script
from akg.topi.nn import pad as tvm_pad
from akg.utils.format_transform import get_shape, get_bytes, to_tvm_const
import akg.utils as  utils
from akg.utils import custom_tiling as ct_util
from akg.utils import dynamic_shape as ds

C_LIMIT_FOR_CAST = 3600


def get_attrs():
    """get attrs."""
    attrs = {
        "help_tiling": 0,
        "pragma_sink_last_axis": False,
        "enable_pre_poly_loop_partition": False
    }
    return attrs


def get_dynamic_attrs():
    """get dynamic attrs."""
    attrs = {
        "help_tiling": 0,
        "pragma_sink_last_axis": False,
        "pragma_disable_whole_component": False,
        "enable_pre_poly_loop_partition": True,
        "dynamic_shape_bound": 65535,
        "enable_post_poly_loop_partition": False,
        "enable_double_buffer:": False,
        # "enable_scalar_align": True,
    }
    return attrs


four2five_set_dim_map = {
    "((1, 1, 7, 7), 'NCHW', 'float32', 'float16')": ((1, 1), (1, 1), (7, 1), (7, 1), (16, 1)),
    "((1, 7, 7), 'NCHW', 'float32', 'float16')": ((1, 1), (7, 1), (7, 1), (16, 1)),
    "((1, 1, I2, I3), 'NCHW', 'float32', 'float16')": ((1, 1), (1, 1), (1, 1), (129, 1), (2048, 1)),
}


def four2five_set_dim_func(data, format_, dst_type):
    """set dim info for attr."""
    shape = get_shape(data)
    if format_ == 'NCHW':
        n, _, h, w = shape
    else:
        n, h, w, _ = shape

    shape[0] = 1
    if h != 1 and w != 1:
        if format_ == 'NCHW' and shape[1] > 16:
            shape[1] = 1
        if format_ == 'NHWC' and shape[-1] > 16:
            shape[-1] = 1

    if n == 1:
        shape.remove(shape[0])

    hash_key = str((tuple(shape), format_, data.dtype, dst_type))
    return ct_util.set_dims_by_key(hash_key, four2five_set_dim_map), hash_key


def four2five_tiling_strategy(tensor, input_format, expansion=None):
    """Custom tiling strategy for four2five op."""
    strategy = ct_util.create_template(tensor=tensor,
                                       template=ct_util.TileTemplate.NC1HWC0)
    if input_format == "NHWC" or expansion:
        priority_map = {4: 0, 1: 1, 3: 2, 2: 3, 0: 4}  # tile in C0->C1->W->H->N sequence
        for pos, priority in priority_map.items():
            strategy.append(ct_util.create_constraint_on_tensor(tensor=tensor,
                                                                values=priority,
                                                                constraints=ct_util.TileConstraint.SET_PRIORITY,
                                                                tensor_pos=pos)[0])
    if expansion:
        strategy.append(ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values=expansion,
                                                            constraints=ct_util.TileConstraint.SET_EXPANSION)[0])
    return strategy

def four2five_tiling_strategy_dynamic(tensor, input_format):
    """Custom tiling strategy for four2five op."""
    strategy = list()
    if input_format == "NCHW":
        shape = get_shape(tensor)
        if shape[1] == 1:
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 1, ct_util.TileConstraint.FACTOR, 0)[0])
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 1, ct_util.TileConstraint.FACTOR, 1)[0])
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 1, ct_util.TileConstraint.FACTOR, 2)[0])
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 112, ct_util.TileConstraint.FACTOR, 3)[0])
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 16, ct_util.TileConstraint.FACTOR, 4)[0])
        elif shape[1] == 128:
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 1, ct_util.TileConstraint.FACTOR, 0)[0])
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 1, ct_util.TileConstraint.FACTOR, 1)[0])
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 1, ct_util.TileConstraint.FACTOR, 2)[0])
            strategy.append(ct_util.create_constraint_on_tensor(tensor, "FULL", ct_util.TileConstraint.MAX, 3)[0])
            strategy.append(ct_util.create_constraint_on_tensor(tensor, 16, ct_util.TileConstraint.FACTOR, 4)[0])
    return strategy

@utils.check_input_type(akg.tvm.tensor.Tensor, str, str, bool, (str, type(None)))
def Four2Five(data, format_, dst_dtype='float16', need_custom_tiling=True, target=utils.CCE):
    """
    Convert 4-dims "data" to 5-dims,the format of "data" is defined in "format_"

    Args:
        data (tvm.tensor.Tensor): 4-dims tensor of type float16, float32
        format_ (str): a str defined the format of "data"
        dst_dtype (str): a str defined the type of output, could be float16 or float32

    Returns:
        5-dims tvm.tensor.Tensor,type is defined by dst_dtype,
        which shape is [N, ceil(C / 16), H, W, 16] and attr about tiling args

    Raises:
        ValueError: If the type of format_ is invalid.
    
    Supported Platforms:
        'Ascend'
    """
    # Check dtype
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    # Check shape
    shape = get_shape(data)
    utils.davinci_format_check(shape, format_, dim=4)

    # Check format
    if format_ not in ['NCHW', 'NHWC']:
        raise ValueError("{} format is not support, four2five only support NCHW and NHWC format input"
                         .format(format_))
    last_channel = 16
    if format_ == "NCHW":
        bs, c, h, w = get_shape(data)
    else:
        bs, h, w, c = get_shape(data)
    pad_c = c
    if c % last_channel != 0:
        pad_c = (c + 15) // last_channel * last_channel
    c1 = pad_c // last_channel
    c0 = last_channel
    is_dynamic = ds.shape_is_dynamic(data)
    if not is_dynamic:
        attrs = get_attrs()
    else:
        attrs = get_dynamic_attrs()
    # Check size c when casting happens
    if data.dtype != dst_dtype and c0 * c1 >= C_LIMIT_FOR_CAST:
        raise ValueError("When input and output data type is not matched, shape of 'c' axis should not exceed {}, "
                         "while currently set is {}".format(C_LIMIT_FOR_CAST, c0 * c1))

    @script(capture=locals())
    def nchw_to_nc1hwc0_step(inputs, bs, c1, h, w, c0):
        output = allocate((bs, c1, h, c0, w), inputs.dtype, "local")
        for n_i in range(bs):
            for c_i in range(c1):
                for h_i in range(h):
                    for w_i in range(w):
                        for c_i0 in range(c0):
                            output[n_i, c_i, h_i, c_i0, w_i] = inputs[n_i, c_i * last_channel + c_i0, h_i, w_i]
        output1 = allocate((bs, c1, h, w, c0), inputs.dtype, "local")
        for n_i in range(bs):
            for c_i in range(c1):
                for h_i in range(h):
                    for w_i in range(w):
                        for c_i0 in range(c0):
                            output1[n_i, c_i, h_i, w_i, c_i0] = output[n_i, c_i, h_i, c_i0, w_i]
        return output1

    @script(capture=locals())
    def nchw_to_nc1hwc0(inputs, bs, c1, h, w, c0):
        output = allocate((bs, c1, h, w, c0), inputs.dtype, "local")
        for n_i in range(bs):
            for c_i in range(c1):
                for h_i in range(h):
                    for w_i in range(w):
                        for c_i0 in range(c0):
                            output[n_i, c_i, h_i, w_i, c_i0] = inputs[n_i, c_i * last_channel + c_i0, h_i, w_i]
        return output

    @script(capture=locals())
    def nhwc_to_nc1hwc0(inputs, zero, bs, c1, h, w, c0):
        output = allocate((bs, c1, h, w, c0), inputs.dtype, "local")
        for n_i in range(bs):
            for c_i in range(c1):
                for h_i in range(h):
                    for w_i in range(w):
                        for c_i0 in range(c0):
                            if c_i * last_channel + c_i0 < c:
                                output[n_i, c_i, h_i, w_i, c_i0] = inputs[n_i, h_i, w_i, c_i * last_channel + c_i0]
                            else:
                                output[n_i, c_i, h_i, w_i, c_i0] = zero

        return output

    cast_data = data
    need_cast = data.dtype == 'float32' and dst_dtype == 'float16'
    if c % last_channel != 0 or need_cast:
        expansion = int(ct_util.BLOCK_SIZE / get_bytes(data.dtype))
    else:
        expansion = None
    # float32 -> float16, need to cast before transform
    if need_cast:
        cast_data = akg.lang.ascend.cast_to(data, dst_dtype)

    zero_ = akg.tvm.const(0.0, cast_data.dtype)
    if format_ == "NCHW":
        if c % last_channel != 0:
            pad_shape = [bs, pad_c, h, w]
            if h == 1 and w == 1:
                # if h and w both are 1, it is pad last dim case
                output_shape = [bs, pad_c // last_channel, h, w, last_channel]

                output = akg.tvm.compute(output_shape,
                                         lambda i, c1, k, l, c0: akg.tvm.expr.Select(
                                             c0 < c - c1 * last_channel, cast_data[i, c1 * last_channel + c0, k, l],
                                             akg.tvm.const(0, cast_data.dtype)),
                                         name="output")
            else:
                # if need to pad c dim, separate transpose to two steps
                # first is nchw -> nc1hc0w, second is nc1hc0w -> nc1hwc0
                pad_data = akg.tvm.compute(pad_shape,
                                           lambda i, j, k, l: akg.tvm.expr.Select(j < c, cast_data[i, j, k, l], zero_),
                                           name="pad_data")
                output = nchw_to_nc1hwc0_step(
                    pad_data,
                    to_tvm_const(bs),
                    to_tvm_const(c1),
                    to_tvm_const(h),
                    to_tvm_const(w),
                    to_tvm_const(c0))

        else:
            if not is_dynamic and data.dtype == "float16" and h * w % last_channel == 0 and h * w < 3600:
                output_shape = [bs, c1, h, w, c0]
                output = akg.tvm.compute(output_shape, lambda n, c1, h, w, c0:
                                         akg.lang.ascend.four2five_nchw(cast_data[n, c1 * last_channel + c0, h, w]),
                                         name="output")

            else:
                output = nchw_to_nc1hwc0(
                    cast_data,
                    to_tvm_const(bs),
                    to_tvm_const(c1),
                    to_tvm_const(h),
                    to_tvm_const(w),
                    to_tvm_const(c0))

    else:
        if not is_dynamic and c < last_channel:
            rank = 5  # (n, c1, h, w, c0)
            pad_before = []
            pad_after = []
            for _ in range(rank):
                pad_before.append(0)
                pad_after.append(0)
            pad_after[-1] = last_channel - c
            # As c < last_channel, c1 is 1
            output = akg.tvm.compute((bs, c1, h, w, c), lambda bs_i, _, h_i, w_i, c_i: cast_data[
                bs_i, h_i, w_i, c_i], name="output")
            output = tvm_pad(output, pad_before, pad_after=pad_after, name='pad_output')
        else:
            output = nhwc_to_nc1hwc0(
                cast_data,
                zero_,
                to_tvm_const(bs),
                to_tvm_const(c1),
                to_tvm_const(h),
                to_tvm_const(w),
                to_tvm_const(c0))

    # float16 -> float32, need to cast after transform
    if data.dtype == 'float16' and dst_dtype == 'float32':
        output = akg.lang.ascend.cast_to(output, dst_dtype)

    utils.davinci_format_check(output.shape, "NC1HWC0", dim=5)

    if not is_dynamic:
        dim_info, _ = four2five_set_dim_func(data, format_, dst_dtype)
        if dim_info != "":
            attrs["dim"] = dim_info
        if need_custom_tiling:
            attrs["custom_tiling"] = four2five_tiling_strategy(output, format_, expansion)
    elif need_custom_tiling:
        attrs["custom_tiling"] = four2five_tiling_strategy_dynamic(output, format_)

    if is_dynamic:
        attrs["enable_feature_library_pre_poly"] = True
    return output, attrs
