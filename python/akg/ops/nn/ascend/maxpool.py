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

"""operator dsl function:maxpool"""
from __future__ import absolute_import
import math
import akg
import akg.tvm
import akg.utils as utils
from akg.tvm.hybrid import script
from akg.utils import custom_tiling as ct_util, kernel_exec as k_utils
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from akg.utils.format_transform import get_shape
from akg.utils import dynamic_shape as ds

maxpool_set_dim_map = {
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float16")):
        ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 0, 1), (0, 4, 0, 1),
         (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 0, 1), (1, 4, 0, 1)),
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', "float16")):
        ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 0, 1), (0, 4, 0, 1),
         (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 0, 1), (1, 4, 0, 1)),
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float32")):
        ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 0, 1), (0, 4, 0, 1),
         (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 0, 1), (1, 4, 0, 1)),
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', "float32")):
        ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 0, 1), (0, 4, 0, 1),
         (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 0, 1), (1, 4, 0, 1)),
    str(((32, 6, 55, 55, 16), (3, 3), (2, 2), 'VALID', "float16")):
        ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 3, 1), (0, 3, 3, 1), (0, 4, 0, 1), (0, 5, 0, 1),
         (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 3, 1), (1, 3, 3, 1), (1, 4, 1, 1), (1, 5, 0, 1)),
    str(((32, 6, 55, 55, 16), (3, 3), (2, 2), 'VALID', "float32")):
        ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 3, 1), (0, 3, 3, 1), (0, 4, 0, 1), (0, 5, 0, 1),
         (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 3, 1), (1, 3, 3, 1), (1, 4, 1, 1), (1, 5, 0, 1)),
}


maxpool_set_attr_map = {
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', "float16")): {
        "merge_outer_loop_for_multicore": 1,
    },
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', "float32")): {
        "merge_outer_loop_for_multicore": 1,
    },
}

attr_map = dict()


def maxpool_set_dim_func(data, kernel, stride, pad):
    """Set dim info with maxpool_set_dim_map."""
    key = []
    key.append(tuple(data.shape))
    key.append(kernel)
    key.append(stride)
    key.append(pad)
    key.append(data.dtype)
    hash_key = str(tuple(key))

    global attr_map
    default_attr_map = {
        "pragma_reorder_schedule": True,
        "pragma_opt_for_dsa": 1,
        "loop_partition_unroll": False,
    }
    attr_map.clear()
    for k, v in default_attr_map.items():
        attr_map[k] = v
    if hash_key in maxpool_set_attr_map.keys():
        attrs_dict = maxpool_set_attr_map.get(hash_key, {})
        for k, v in attrs_dict.items():
            attr_map[k] = v

    if hash_key in maxpool_set_dim_map.keys():
        return ct_util.set_dims(maxpool_set_dim_map.get(hash_key, {})), hash_key
    return "", hash_key


def maxpool_param_check(kernel, stride, pad):
    """check maxpool parameters"""
    if len(kernel) != 2:
        raise ValueError("Only support 2-dim kernel!")
    if len(stride) != 2:
        raise ValueError("Only support 2-dim stride!")
    if len(pad) != 2 and (len(pad) != 4 or pad[0] != pad[1] or pad[2] != pad[3]):
        raise ValueError(
            "Only support 2-dim pad, or 4-dim with 2 equal values!")


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple),
                        (list, tuple), (list, tuple))
def old_maxpool(data, kernel, stride, pad):
    """
    Old implement for maxpool.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16 or float32, \"NC1HWC0\"
                                  format (N: batch, C1: channel, H: height, W:
                                  width, C0: block size)
        kernel (Union[list, tuple]): List or tuple with two int number as
                                     window sizes of H and W.
        stride (Union[list, tuple]): List or tuple with two int number as
                                     stride sizes of H and W.
        pad (Union[list, tuple]): List or tuple with two int number as
                                  pad sizes of H and W.

    Returns:
        tvm.tensor.Tensor, result of maxpool operator.
    """
    shape = get_shape(data)
    dtype = data.dtype
    utils.davinci_format_check(shape, "NC1HWC0", dim=5)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)

    maxpool_param_check(kernel, stride, pad)

    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    if len(pad) == 2:
        pad_height, pad_width = pad
    else:
        pad_height, pad_width = pad[0], pad[2]

    in_n, in_c1, in_h, in_w, in_c0 = shape

    out_h = int(math.floor((in_h + 2 * pad_height - kernel_h)
                           / float(stride_h)) + 1)
    out_w = int(math.floor((in_w + 2 * pad_width - kernel_w)
                           / float(stride_w)) + 1)

    if pad[0] != 0 or pad[1] != 0:
        pad_shape = (in_n, in_c1, in_h + 2 * pad_height,
                     in_w + 2 * pad_width, in_c0)

        pad2d = akg.tvm.compute(pad_shape,
                                lambda n, c1, h, w, c0:
                                akg.tvm.const(0.0, dtype=dtype),
                                name="pad2d"
                                )
        pad2d = akg.tvm.compute(pad_shape,
                                lambda n, c1, h, w, c0:
                                akg.tvm.if_then_else(
                                    akg.tvm.any(
                                        h < pad_height,
                                        h > in_h + pad_height - 1,
                                        w < pad_width,
                                        w > in_w + pad_width - 1
                                    ),
                                    pad2d[n, c1, h, w, c0],
                                    data[n, c1, h - pad_height,
                                         w - pad_width, c0],
                                ),
                                name="pad2d"
                                )
    else:
        pad2d = data

    axis_kernel_h = akg.tvm.reduce_axis((0, kernel_h), name="ah")
    axis_kernel_w = akg.tvm.reduce_axis((0, kernel_w), name="aw")

    out_shape = (in_n, in_c1, out_h, out_w, in_c0)

    res_value = akg.tvm.compute(out_shape,
                                lambda n, c1, h, w, c0:
                                akg.tvm.max(
                                    pad2d[n, c1, h * stride_h + axis_kernel_h,
                                          w * stride_w + axis_kernel_w, c0],
                                    axis=[axis_kernel_h, axis_kernel_w]
                                ),
                                name="res_value")
    return res_value


def maxpool_manual_schedule(shape, kernel, stride, padding, dtype, attrs=None, polyhedral=False):
    """maxpool with manual schedule"""
    utils.davinci_format_check(shape, "NC1HWC0", dim=5)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)

    maxpool_param_check(kernel, stride, padding)

    data = akg.tvm.placeholder(shape, dtype, name="input_data")
    batch_size, in_c1, input_h, input_w, in_c0 = data.shape

    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    if len(padding) == 2:
        pad_h, pad_w = padding
    elif len(padding) == 4:
        pad_h, pad_w = padding[0], padding[2]

    out_size_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_size_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1

    # padding operation
    if pad_h != 0 or pad_w != 0:
        pad_shape = (batch_size, in_c1, input_h + 2 *
                     pad_h, input_w + 2 * pad_w, in_c0)

        padded_input = akg.tvm.compute(pad_shape,
                                       lambda n, c1, h, w, c0:
                                       akg.tvm.if_then_else(
                                           akg.tvm.any(
                                               h > input_h + pad_h - 1,
                                               h < pad_h,
                                               w > input_w + pad_w - 1,
                                               w < pad_w,
                                           ),
                                           akg.tvm.const(0.0, dtype=dtype),
                                           data[n, c1, h - pad_h,
                                                w - pad_w, c0],
                                       ),
                                       name="padded_input")
    else:
        padded_input = data

    # reduce iterators
    it_kernel_h = akg.tvm.reduce_axis(
        (0, kernel_h), name="iterator_reduction_height")
    it_kernel_w = akg.tvm.reduce_axis(
        (0, kernel_w), name="iterator_reduction_width")

    out_shape = (batch_size, in_c1, out_size_h, out_size_w, in_c0)

    res = akg.tvm.compute(out_shape,
                          lambda n, c1, h, w, c0:
                          akg.tvm.max(
                              padded_input[n, c1, (h * stride_h + it_kernel_h),
                                           (w * stride_w + it_kernel_w), c0],
                              axis=[it_kernel_h, it_kernel_w]
                          ),
                          name="maxpool_not_hybrid")

    s = akg.tvm.create_schedule([res.op])

    if pad_w != 0 or pad_h != 0:
        padded_input = res.op.input_tensors[0]
    else:
        padded_input = res

    # cache reads and writes
    # after this cache write: reference to res_ub to change the reduction axis
    res_ub = s.cache_write(res, "local.UB")
    if pad_w != 0 or pad_h != 0:
        data_ub = s.cache_read(data, "local.UB", [padded_input])
    else:
        data_ub = s.cache_read(data, "local.UB", [res_ub])

    # get tiling attributes
    if attrs is None:
        raise Exception('attrs is None')
    tiling_factors = attrs['tile']
    split_iterators = []
    if len(tiling_factors) != len(res.shape):
        raise RuntimeError("tiling factors mismatch out shape")
    # split the final compute and save the iterators
    for index, factor in enumerate(tiling_factors):
        split_iterators.append(s[res_ub].split(res_ub.op.axis[index], factor))

    # get iterators
    iterator_b_outer = split_iterators[0][0]
    iterator_b_inner = split_iterators[0][1]
    iterator_c1_outer = split_iterators[1][0]
    iterator_c1_inner = split_iterators[1][1]
    iterator_h_outer = split_iterators[2][0]
    iterator_h_inner = split_iterators[2][1]
    iterator_w_outer = split_iterators[3][0]
    iterator_w_inner = split_iterators[3][1]
    iterator_c0_outer = split_iterators[4][0]
    iterator_c0_inner = split_iterators[4][1]
    # reduction axis
    iterator_reduce_h = res_ub.op.reduce_axis[0]
    iterator_reduce_w = res_ub.op.reduce_axis[1]

    # move caches
    s[res_ub].compute_at(s[res], res.op.axis[0])
    s[data_ub].compute_at(s[res_ub], iterator_c1_outer)

    if pad_w != 0 or pad_h != 0:
        s[padded_input].compute_at(s[res_ub], iterator_c1_outer)
        s[padded_input].set_scope("local.UB")

    # reorder computation
    s[res_ub].reorder(iterator_b_outer, iterator_b_inner, iterator_c1_outer, iterator_c1_inner, iterator_h_outer,
                      iterator_h_inner, iterator_w_outer, iterator_w_inner, iterator_reduce_h, iterator_reduce_w,
                      iterator_c0_outer, iterator_c0_inner)

    with akg.build_config(add_lower_pass=k_utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [data, res], "cce", name="maxpool_manual_schedule",
                        attrs=attrs, polyhedral=polyhedral)
        source_code = mod.imported_modules[0].get_source()
        kernel_name = "maxpool_ad_manual_schedule"
        k_utils.create_code(kernel_name, './', source_code)
    return mod


def pad_strategy_check(strategy):
    """check the correctness of pad strategy"""
    if not isinstance(strategy, str) \
            and not (isinstance(strategy, (list, tuple)) and len(strategy) == 4):
        raise ValueError("Only support string or list/tuple of 4 int numbers!")


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), (list, tuple),
                        (list, tuple, str), (str, type(None)))
def maxpool(data, kernel, stride, strategy):
    """
    Performs the max pooling on the input data.

    Note:
        Only support 5D format(NC1HWC0), and pooling will work on H and W.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.
        kernel (Union[list, tuple]): two int numbers for pooling window's size.
        stride (Union[list, tuple]): two int numbers for window's stride.
        strategy (Union[str, list, tuple]): padding, should be 'VALID',
            'SAME' or instance of list(four int numbers for 'CONSTANTS' strategy).
            Support **Strategies** is same as avgpool.

    Returns:
        tvm.tensor.Tensor, as result for max pooling.

    Supported Platforms:
        'Ascend'
    """
    attrs = attr_map
    attrs['dim'] = maxpool_set_dim_func(data, kernel, stride, strategy)[0]

    shape = get_shape(data)
    dtype = data.dtype
    utils.davinci_format_check(shape, "NC1HWC0", dim=5)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(kernel, 2, "Kernel")
    utils.check_shape(stride, 2, "Stride")

    pad_strategy_check(strategy)

    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    in_n, in_c1, in_h, in_w, in_c0 = shape

    [ph_h, _, pw_h, _], [out_h, out_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, strategy)
    if attrs.get("dynamic") is True:
        # dynamic shape: although we can represent out_h and out_w using input shapes, they are too complicated
        out_h = akg.tvm.var("OUT_H")
        out_w = akg.tvm.var("OUT_W")

    @script(capture=locals())
    def dynamic_max_pool_hybrid_0(min_value_, x_, in_n, in_c1, in_h, in_w, in_c0, out_h, out_w):
        output = output_tensor((in_n, in_c1, out_h, out_w, in_c0), x_.dtype)

        for n in range(in_n):
            for c1 in range(in_c1):
                # Head
                for ow in range(out_w):
                    for c0 in range(in_c0):
                        output[n, c1, 0, ow, c0] = min_value_
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        for ow in range(out_w):
                            for c0 in range(in_c0):
                                if ph_h <= kh <= in_h + ph_h - 1 and 0 <= ow * stride_w + kw - pw_h <= in_w - 1:
                                    output[n, c1, 0, ow, c0] = \
                                        max(output[n, c1, 0, ow, c0],
                                            x_[n, c1, kh - ph_h, ow * stride_w + kw - pw_h, c0])
                # Tail
                for oh in range(out_h - 1):
                    for ow in range(out_w):
                        for c0 in range(in_c0):
                            output[n, c1, oh + 1, ow, c0] = min_value_
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        for oh in range(out_h - 1):
                            for ow in range(out_w):
                                for c0 in range(in_c0):
                                    if ph_h <= (oh + 1) * stride_h + kh <= in_h + ph_h - 1\
                                            and pw_h <= ow * stride_w + kw <= in_w + pw_h - 1:
                                        output[n, c1, oh + 1, ow, c0] = max(output[n, c1, oh + 1, ow, c0],
                                                                            x_[n, c1, (oh + 1) * stride_h
                                                                               + kh - ph_h, ow * stride_w
                                                                               + kw - pw_h, c0])

        return output

    # static shape's hybrid
    @script(capture=locals())
    def static_max_pool_hybrid_0(min_value_, x_):
        output = output_tensor((in_n, in_c1, out_h, out_w, in_c0), x_.dtype)

        for n in range(in_n):
            for c1 in range(in_c1):
                # Head
                for ow in range(out_w):
                    for c0 in range(in_c0):
                        output[n, c1, 0, ow, c0] = min_value_
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        for ow in range(out_w):
                            for c0 in range(in_c0):
                                if ph_h <= kh <= in_h + ph_h - 1 and 0 <= ow * stride_w + kw - pw_h <= in_w - 1:
                                    output[n, c1, 0, ow, c0] = \
                                        max(output[n, c1, 0, ow, c0],
                                            x_[n, c1, kh - ph_h, ow * stride_w + kw - pw_h, c0])
                # Tail
                for oh in range(out_h - 1):
                    for ow in range(out_w):
                        for c0 in range(in_c0):
                            output[n, c1, oh + 1, ow, c0] = min_value_
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        for oh in range(out_h - 1):
                            for ow in range(out_w):
                                for c0 in range(in_c0):
                                    if ph_h <= (oh + 1) * stride_h + kh <= in_h + ph_h - 1 \
                                            and pw_h <= ow * stride_w + kw <= in_w + pw_h - 1:
                                        output[n, c1, oh + 1, ow, c0] = max(output[n, c1, oh + 1, ow, c0],
                                                                            x_[n, c1, (oh + 1) * stride_h
                                                                               + kh - ph_h, ow * stride_w
                                                                               + kw - pw_h, c0])

        return output

    min_value = akg.tvm.const(-65504.0 if dtype == 'float16'
                              else -340282346638528859811704183484516925440.0, dtype=dtype)
    if attrs.get("dynamic") is True:
        output = dynamic_max_pool_hybrid_0(min_value, data,
                                           in_n, in_c1, in_h, in_w, in_c0, out_h, out_w)
    else:
        output = static_max_pool_hybrid_0(min_value, data)

    return output, attrs


maxpool_with_argmax_set_dim_map = {
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', "float16")):
        ((0, 0, 1, 1), (0, 1, 1, 1), (0, "H", 3, 1),
         (0, 3, 56, 1), (0, 4, 5, 1), (0, 5, 16, 1)),
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), "float16")):
        ((0, 0, 1, 1), (0, 1, 1, 1), (0, "H", 3, 1),
         (0, 3, 56, 1), (0, 4, 5, 1), (0, 5, 16, 1)),
    str(((1, 1, 28, 28, 16), (2, 2), (2, 2), 'VALID', "float16")):
        ((0, 0, 14, 1), (0, 1, 14, 1), (0, 4, 3, 1), (0, 5, 16, 1)),
    # str('((I0, I1, I2, I3, 16), (3, 3), (2, 2), (1, 1, 1, 1), \'float16\')'):
    #     ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 3, 1), (0, 3, 3, 1), (0, 4, 3, 1),
    #      (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 3, 1), (1, 3, 3, 1), (1, "H", 3, 1)),
}

maxpool_with_argmax_set_attr_map = {
}
attr_map_v2 = dict()


def maxpool_with_argmax_tiling_strategy(data, kernel, stride, pad):
    """Custom tiling for maxpool with argmax version."""
    batch, c1, fm_h, fm_w, c0 = data.shape
    _, [out_h, _] = \
        cal_pad_shapes_by_strategy(get_shape(data), kernel, stride, pad)
    strategy = list()
    if data.ndim == 5 and c0.value == 16:
        h_cut = out_h
        if isinstance(fm_h, akg.tvm.expr.Var) or (fm_h.value >= 50 and fm_w.value >= 50):
            h_cut = 3
        dim_ind = 0
        if isinstance(batch, akg.tvm.expr.Var) or batch.value > 1:
            strategy += ct_util.create_constraint_on_axis(values=1,
                                                          constraints=ct_util.TileConstraint.FACTOR,
                                                          axis=dim_ind)
            dim_ind = dim_ind + 1
        if isinstance(c1, akg.tvm.expr.Var) or c1.value > 1:
            strategy += ct_util.create_constraint_on_axis(values=1,
                                                          constraints=ct_util.TileConstraint.FACTOR,
                                                          axis=dim_ind)
            dim_ind = dim_ind + 1
        strategy += ct_util.create_constraint_on_axis(values=h_cut,
                                                      constraints=ct_util.TileConstraint.FACTOR,
                                                      axis=dim_ind)
        strategy += ct_util.create_constraint_on_axis(values="H",
                                                      constraints=ct_util.TileConstraint.SET_AXIS_INFO,
                                                      axis=dim_ind)
        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      axis=dim_ind + 1)
        strategy += ct_util.create_constraint_on_axis(values=5,
                                                      constraints=ct_util.TileConstraint.FACTOR,
                                                      axis=dim_ind + 2)
        strategy += ct_util.create_constraint_on_axis(values=16,
                                                      constraints=ct_util.TileConstraint.FACTOR,
                                                      axis=dim_ind + 3)
    return strategy


def maxpool_with_argmax_dynamic_tensor_strategy(data, im2col, mask):
    """Custom tiling for maxpool with argmax version."""
    _, _, _, _, c0 = data.shape
    strategy = list()
    if data.ndim == 5 and c0.value == 16:
        strategy += ct_util.create_constraint_on_tensor(tensor=im2col,
                                                        values=1,
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=0)
        strategy += ct_util.create_constraint_on_tensor(tensor=im2col,
                                                        values=1,
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=1)
        strategy += ct_util.create_constraint_on_tensor(tensor=im2col,
                                                        values="FULL",
                                                        constraints=ct_util.TileConstraint.MAX,
                                                        tensor_pos=2)
        strategy += ct_util.create_constraint_on_tensor(tensor=im2col,
                                                        values="FULL",
                                                        constraints=ct_util.TileConstraint.MAX,
                                                        tensor_pos=3)
        strategy += ct_util.create_constraint_on_tensor(tensor=im2col,
                                                        values=1,
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=4)
        strategy += ct_util.create_constraint_on_tensor(tensor=im2col,
                                                        values="FULL",
                                                        constraints=ct_util.TileConstraint.MAX,
                                                        tensor_pos=5)
        strategy += ct_util.create_constraint_on_tensor(tensor=im2col,
                                                        values="FULL",
                                                        constraints=ct_util.TileConstraint.MAX,
                                                        tensor_pos=6)

        strategy += ct_util.create_constraint_on_tensor(tensor=mask,
                                                        values=1,
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=0)
        strategy += ct_util.create_constraint_on_tensor(tensor=mask,
                                                        values=1,
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=1)
        strategy += ct_util.create_constraint_on_tensor(tensor=mask,
                                                        values=1,
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=2)
        strategy += ct_util.create_constraint_on_tensor(tensor=mask,
                                                        values=1,
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=3)
        strategy += ct_util.create_constraint_on_tensor(tensor=mask,
                                                        values="FULL",
                                                        constraints=ct_util.TileConstraint.MAX,
                                                        tensor_pos=4)
        strategy += ct_util.create_constraint_on_tensor(tensor=mask,
                                                        values="FULL",
                                                        constraints=ct_util.TileConstraint.MAX,
                                                        tensor_pos=5)
        strategy += ct_util.create_constraint_on_tensor(tensor=mask,
                                                        values="FULL",
                                                        constraints=ct_util.TileConstraint.MAX,
                                                        tensor_pos=6)
    return strategy


def maxpool_with_argmax_custom_tiling_strategy(data):
    """Custom tiling for maxpool with argmax version."""
    batch, c1, _, _, c0 = data.shape
    strategy = list()
    if data.ndim == 5 and c0.value == 16:
        band = 1
        dim_ind = 0
        if isinstance(batch, akg.tvm.expr.Var) or batch.value > 1:
            strategy += ct_util.create_constraint_on_axis(values=1,
                                                          constraints=ct_util.TileConstraint.FACTOR,
                                                          band=band,
                                                          axis=dim_ind)
            dim_ind = dim_ind + 1
        if isinstance(c1, akg.tvm.expr.Var) or c1.value > 1:
            strategy += ct_util.create_constraint_on_axis(values=1,
                                                          constraints=ct_util.TileConstraint.FACTOR,
                                                          band=band,
                                                          axis=dim_ind)
            dim_ind = dim_ind + 1
        strategy += ct_util.create_constraint_on_axis(values=1,
                                                      constraints=ct_util.TileConstraint.FACTOR,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1
        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1
        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1
        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1
        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
        band = 0
        dim_ind = 0
        strategy += ct_util.create_constraint_on_axis(values=1,
                                                      constraints=ct_util.TileConstraint.FACTOR,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1
        strategy += ct_util.create_constraint_on_axis(values=1,
                                                      constraints=ct_util.TileConstraint.FACTOR,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1

        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1

        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1

        strategy += ct_util.create_constraint_on_axis(values=1,
                                                      constraints=ct_util.TileConstraint.FACTOR,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1

        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1

        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
        dim_ind = dim_ind + 1
        strategy += ct_util.create_constraint_on_axis(values="FULL",
                                                      constraints=ct_util.TileConstraint.MAX,
                                                      band=band,
                                                      axis=dim_ind)
    return strategy


def get_attrs():
    """Get default attrs for maxpool."""
    default_attr_map = {
        "pragma_opt_for_dsa": 1,
        "pragma_reorder_schedule": True,
        "enable_pre_poly_loop_partition": False,
        "enable_post_poly_loop_partition": False,
        "disable_cse": True,
        "enable_bk_optimize": False,
        "enable_to_three_address": False
    }
    return default_attr_map


def get_dynamic_attrs():
    """Get default attrs for maxpool."""
    default_attr_map = {
        "pragma_opt_for_dsa": 1,
        "pragma_reorder_schedule": True,
        "enable_pre_poly_loop_partition": False,
        "enable_post_poly_loop_partition": False,
        "disable_cse": True,
        "enable_bk_optimize": False,
        "enable_double_buffer": False,
        "enable_hoist_cond_write": False,
        "extent_to_cond": False,
        "merge_outer_loop_for_multicore": 1,
        "multicore_loop_max_depth": 2,
        "enable_sink_allocate": True,
    }
    return default_attr_map


def maxpool_with_argmax_set_dim_func(data, kernel, stride, pad):
    """set dim info for attr"""
    key = []
    key.append(tuple(data.shape))
    key.append(tuple(kernel))
    key.append(tuple(stride))
    if isinstance(pad, list):
        pad = tuple(pad)
    elif isinstance(pad, str):
        pad = pad.upper()
    key.append(pad)
    key.append(data.dtype)
    hash_key = str(tuple(key))

    global attr_map_v2
    default_attr_map = get_attrs()
    attr_map_v2.clear()
    for k, v in default_attr_map.items():
        attr_map_v2[k] = v
    if hash_key in maxpool_with_argmax_set_attr_map.keys():
        attrs_dict = maxpool_with_argmax_set_attr_map.get(hash_key, {})
        for k, v in attrs_dict.items():
            attr_map_v2[k] = v

    if hash_key in maxpool_with_argmax_set_dim_map.keys():
        return ct_util.set_dims(maxpool_with_argmax_set_dim_map.get(hash_key, {})), hash_key
    return "", hash_key


def maxpool_value(index):
    """return index int of maxpool"""
    print(type(index))
    if isinstance(index, akg.tvm.expr.IntImm):
        return index.value
    return index


def img2col(input_img, col_shape, filter_h, filter_w, pad, stride, min_value, tag=None):
    """implement ima2col"""
    def img2col_compute(input_img, indices, filter_w, pad, stride):
        _, _, fmap_h, fmap_w, _ = input_img.shape
        col_n, col_c1, col_hw, col_ww, col_ho, col_wo, col_c0 = indices
        stride_h, stride_w = stride
        pad_top, pad_bottom, pad_left, pad_right = pad

        img_n_index = col_n
        img_c1_index = col_c1
        img_h_index = col_ho * stride_h + col_hw
        img_w_index = col_wo * stride_w + col_ww
        img_c0_index = col_c0
        dilation_h = 1
        dilation_w = 1
        repeat_mode = 1
        jmp_offset = 1

        return akg.lang.ascend.load_im2col_c1_buf(
            akg.tvm.if_then_else(
                akg.tvm.any(
                    img_h_index < pad_top,
                    img_h_index > maxpool_value(fmap_h) + pad_top - 1,
                    img_w_index < pad_left,
                    img_w_index > maxpool_value(fmap_w) + pad_left - 1),
                min_value,
                input_img(
                    img_n_index,
                    img_c1_index,
                    img_h_index - pad_top,
                    img_w_index - pad_left,
                    img_c0_index)),
            pad_top, pad_bottom, pad_left, pad_right, fmap_h, fmap_w,
            stride_h, stride_w, filter_h, filter_w, dilation_h, dilation_w,
            repeat_mode, jmp_offset)

    if tag is None:
        tag = 'im2col_row_major'
    return akg.tvm.compute(
        col_shape,
        lambda *indices: img2col_compute(input_img,
                                         indices, filter_w, pad, stride),
        name='im2col_row_major',
        tag=tag,
        attrs={
            'pragma_conv_kernel_h': filter_h,
            'pragma_conv_kernel_w': filter_w,
            'pragma_conv_padding_top': pad[0],
            'pragma_conv_padding_bottom': pad[1],
            'pragma_conv_padding_left': pad[2],
            'pragma_conv_padding_right': pad[3],
            'pragma_conv_stride_h': stride[0],
            'pragma_conv_stride_w': stride[1],
            'pragma_conv_dilation_h': 1,
            'pragma_conv_dilation_w': 1,
            'pragma_conv_fm_h': input_img.shape[2],
            'pragma_conv_fm_w': input_img.shape[3],
            'pragma_conv_h_cut': (3 - 1) * stride[0] + filter_h,
            'pragma_conv_w_cut': input_img.shape[3]
        })


@ct_util.reg_set_dim_func(maxpool_with_argmax_set_dim_func)
@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), (list, tuple), (list, tuple, str))
def maxpool_with_argmax_dynamic(data, kernel, stride, strategy):
    """
    Performs the max pooling on the input datas.

    Note:
        Only support 5D format(NC1HWC0), and pooling will work on H and W.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.
        kernel (Union[list, tuple]): two int numbers for pooling window's size.
        stride (Union[list, tuple]): two int numbers for window's stride.
        strategy (Union[str, list, tuple]): padding, should be 'VALID','SAME' or
            instance of list(four int numbers, as 'CONSTANTS' strategy).
            Support **Strategies** is the same as avgpool.

    Returns:
        tvm.tensor.Tensor, result for gradient of maxpooling.
    """
    attrs = get_dynamic_attrs()
    dim_info = maxpool_with_argmax_set_dim_func(
        data, kernel, stride, strategy)[0]
    for k, v in attr_map_v2.items():
        attrs[k] = v
    if dim_info != "":
        attrs['dim'] = dim_info
    attrs["enable_feature_library"] = True
    shape = get_shape(data)
    dtype = data.dtype

    utils.davinci_format_check(shape, "NC1HWC0", dim=5)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.FLOAT16)
    utils.check_shape(kernel, 2, 'Kernel')
    utils.check_shape(stride, 2, 'Stride')

    pad_strategy_check(strategy)

    kernel_h, kernel_w = kernel
    in_n, in_c1, _, _, in_c0 = shape

    [ph_h, ph_t, pw_h, pw_t], [out_h, out_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, strategy)

    pad = [ph_h, ph_t, pw_h, pw_t]
    zero = akg.tvm.const(0.0, dtype=dtype)
    min_value = akg.tvm.const(-65504.0 if dtype == 'float16'
                              else -340282346638528859811704183484516925440.0, dtype=dtype)

    # fmap img2col l1 -> ub in zZ format by fractal
    fmap_img2col_shape_ub = (in_n, in_c1, kernel_h,
                             kernel_w, out_h, out_w, in_c0)

    fmap_img2col_ub = img2col(data, fmap_img2col_shape_ub, kernel_h, kernel_w,
                              pad, stride, min_value, tag='')

    out_shape = (in_n, in_c1, out_h, out_w, in_c0)
    reduce_axis_h = akg.tvm.reduce_axis((0, kernel_h), name="reduce_h")
    reduce_axis_w = akg.tvm.reduce_axis((0, kernel_w), name="reduce_w")
    output = akg.tvm.compute(out_shape,
                             lambda n, c1, oh, ow, c0:
                             akg.tvm.max(
                                 fmap_img2col_ub[n, c1, reduce_axis_h,
                                                 reduce_axis_w, oh, ow, c0],
                                 axis=[reduce_axis_h, reduce_axis_w]),
                             name="pooling_max")

    zero = akg.tvm.const(0.0, dtype=dtype)
    mask_first_max_shape = (in_n, in_c1, kernel_h,
                            kernel_w, out_h, out_w, in_c0)
    mask_first_max = akg.tvm.compute(
        mask_first_max_shape, lambda *indice: zero, name="mask_first_max")

    attrs["custom_tiling"] = maxpool_with_argmax_dynamic_tensor_strategy(
        data, fmap_img2col_ub, mask_first_max)
    attrs["dynamic_shape"] = ds.set_dynamic_shape_limit_for_tensor(output, [
                                                                   64, 64], [2, 3])
    return output, mask_first_max, attrs


@ct_util.reg_set_dim_func(maxpool_with_argmax_set_dim_func)
@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), (list, tuple), (list, tuple, str), (str, type(None)))
def maxpool_with_argmax(data, kernel, stride, strategy):
    """
    Performs the max pooling on the input datas.

    Note:
        Only support 5D format(NC1HWC0), and pooling will work on H and W.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.
        kernel (Union[list, tuple]): two int numbers for pooling window's size.
        stride (Union[list, tuple]): two int numbers for window's stride.
        strategy (Union[str, list, tuple]): padding, should be 'VALID','SAME' or
            instance of list(four int numbers, as 'CONSTANTS' strategy).
            Support **Strategies** is the same as avgpool.

    Returns:
        tvm.tensor.Tensor, result for gradient of maxpooling.
    """
    attrs = get_attrs()
    dim_info = maxpool_with_argmax_set_dim_func(
        data, kernel, stride, strategy)[0]
    for k, v in attr_map_v2.items():
        attrs[k] = v
    if dim_info != "":
        attrs['dim'] = dim_info
    attrs["custom_tiling"] = maxpool_with_argmax_tiling_strategy(
        data, kernel, stride, strategy)
    shape = get_shape(data)
    dtype = data.dtype

    utils.davinci_format_check(shape, "NC1HWC0", dim=5)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.FLOAT16)
    utils.check_shape(kernel, 2, 'Kernel')
    utils.check_shape(stride, 2, 'Stride')

    pad_strategy_check(strategy)

    kernel_h, kernel_w = kernel
    in_n, in_c1, _, _, in_c0 = shape

    [ph_h, ph_t, pw_h, pw_t], [out_h, out_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, strategy)

    pad = [ph_h, ph_t, pw_h, pw_t]
    zero = akg.tvm.const(0.0, dtype=dtype)
    one = akg.tvm.const(1.0, dtype=dtype)
    min_value = akg.tvm.const(-65504.0 if dtype == 'float16'
                              else -340282346638528859811704183484516925440.0, dtype=dtype)

    # fmap img2col l1 -> ub in zZ format by fractal
    fmap_img2col_shape_ub = (in_n, in_c1, kernel_h,
                             kernel_w, out_h, out_w, in_c0)

    fmap_img2col_ub = img2col(data, fmap_img2col_shape_ub, kernel_h, kernel_w,
                              pad, stride, min_value, tag='')

    out_shape = (in_n, in_c1, out_h, out_w, in_c0)
    reduce_axis_h = akg.tvm.reduce_axis((0, kernel_h), name="reduce_h")
    reduce_axis_w = akg.tvm.reduce_axis((0, kernel_w), name="reduce_w")
    output = akg.tvm.compute(out_shape,
                             lambda n, c1, oh, ow, c0:
                             akg.tvm.max(
                                 fmap_img2col_ub[n, c1, reduce_axis_h,
                                                 reduce_axis_w, oh, ow, c0],
                                 axis=[reduce_axis_h, reduce_axis_w]),
                             name="pooling_max")

    pooling_mask = akg.tvm.compute(fmap_img2col_shape_ub,
                                   lambda n, c1, kh, kw, oh, ow, c0:
                                   akg.tvm.if_then_else(
                                       fmap_img2col_ub[n, c1,
                                                       kh, kw, oh, ow, c0]
                                       < output[n, c1, oh, ow, c0], zero, one),
                                   name="pooling_mask")

    mask_flag = akg.tvm.compute(
        out_shape,
        lambda n, c1, oh, ow, c0: pooling_mask[n, c1, 0, 0, oh, ow, c0],
        name="mask_flag")

    mask_init = akg.tvm.compute(
        out_shape,
        lambda n, c1, oh, ow, c0: pooling_mask[n, c1, 0, 0, oh, ow, c0],
        name="mask_init")

    # spec 2
    @script(capture=locals())
    def hybrid_first_max(mask_, flag_, flag2_, zero_):
        output_ = allocate((in_n, in_c1, kernel_h, kernel_w,
                           out_h, out_w, in_c0), mask_.dtype, 'local')
        for n_i in range(in_n):
            for c1_i in range(in_c1):
                for oh_i in range(out_h):
                    for ow_i in range(out_w):
                        for c0_i in range(in_c0):
                            output_[n_i, c1_i, 0, 0, oh_i, ow_i, c0_i] = flag2_[
                                n_i, c1_i, oh_i, ow_i, c0_i]
                for kh_i in range(kernel_h):
                    for kw_i in range(kernel_w):
                        for oh_i in range(out_h):
                            for ow_i in range(out_w):
                                for c0_i in range(in_c0):
                                    output_[n_i, c1_i, kh_i, kw_i, oh_i, ow_i, c0_i] = \
                                        mask_[n_i, c1_i, kh_i, kw_i, oh_i, ow_i, c0_i] -\
                                        flag_[n_i, c1_i, oh_i, ow_i, c0_i]
                                    output_[n_i, c1_i, kh_i, kw_i, oh_i, ow_i, c0_i] = \
                                        max(output_[
                                            n_i, c1_i, kh_i, kw_i, oh_i, ow_i, c0_i], zero_)
                                    flag_[n_i, c1_i, oh_i, ow_i, c0_i] =\
                                        flag_[n_i, c1_i, oh_i, ow_i, c0_i] +\
                                        output_[n_i, c1_i, kh_i,
                                                kw_i, oh_i, ow_i, c0_i]
        return output_

    mask_first_max = hybrid_first_max(
        pooling_mask, mask_flag, mask_init, zero)
    return output, mask_first_max, attrs
