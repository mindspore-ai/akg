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

"""operator dsl function: maxpool_ad"""
import akg.tvm
import akg.topi
import akg
import akg.utils as utils
from akg.ops.nn.ascend.maxpool import old_maxpool
from akg.utils.format_transform import get_shape
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from akg.utils.kernel_exec import debug_mode, create_code


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (list, tuple), (list, tuple),
                        (str, list, tuple))
def maxpool_ad_no_custom_diff_poly_all_max(head, data, kernel, stride, pad):
    """automatic differentiate of maxpool with polyhedral"""
    attrs = {"enable_post_poly_loop_partition": False,
             "enable_pre_poly_loop_partition": False}
    maxpool_fwd = old_maxpool(data, kernel, stride, pad)
    [dl_ddata] = akg.differentiate(maxpool_fwd, [data], head, None, None)
    return dl_ddata, attrs


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        (list, tuple), (list, tuple), (str, list, tuple))
def maxpool_ad_no_custom_diff_manual_schedule_all_max(head, data, kernel, stride, pad):
    """automatic differentiate of maxpool with manual schedule."""
    attrs = {"enable_post_poly_loop_partition": False,
             "enable_pre_poly_loop_partition": False}
    maxpool_fwd = old_maxpool(data, kernel, stride, pad)
    [dl_ddata] = akg.differentiate(maxpool_fwd, [data], head, None, None)

    # schedule for differetiation operation
    s = akg.tvm.create_schedule([dl_ddata.op])

    new_tensor_red = dl_ddata
    new_tensor = new_tensor_red.op.input_tensors[0]
    data = new_tensor.op.input_tensors[0]
    broadcast = new_tensor.op.input_tensors[1]
    head = new_tensor.op.input_tensors[2]
    forward = broadcast.op.input_tensors[0]

    def comp_func(s):
        data_ub = s.cache_read(data, "local.UB", [forward, new_tensor])
        head_ub = s.cache_read(head, "local.UB", [new_tensor])        
        result_ub = s.cache_write(new_tensor_red, "local.UB")

        s[broadcast].set_scope("local.UB")
        s[forward].set_scope("local.UB")

        b, c1, h, w, c0 = forward.op.axis
        oh, ow = forward.op.reduce_axis
        s[forward].reorder(oh, ow, b, c1, h, w, c0)
        s[new_tensor].set_scope("local.UB")

        b, c1, h, w, c0 = result_ub.op.axis
        s[result_ub].reorder(*result_ub.op.reduce_axis, b, c1, h, w, c0)

        s[broadcast].compute_at(s[result_ub], b)
        s[new_tensor].compute_at(s[result_ub], b)

    return dl_ddata, comp_func, attrs


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        akg.tvm.tensor.Tensor, (list, tuple), (list, tuple), (str, list, tuple), (str, type(None)))
def maxpool_ad(head, data, forward, mask, kernel, stride, pad, target=utils.CCE):
    """
    automatic differentiate of maxpool with manual schedule.

    Supported Platforms:
        'Ascend'
    """
    shape = get_shape(data)
    dtype = data.dtype

    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    [ph_h, _, pw_h, _], [out_size_h, out_size_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, pad)
    batch_size, input_c1, input_h, input_w, input_c0 = shape

    # tile size one is proved to be the most efficient one
    tile_scale_h = 1
    tile_scale_w = 1

    tile_h = stride_h * tile_scale_h

    if kernel_h == stride_h:  # non-overlapping case
        tile_h_pad_u = ph_h % stride_h
    elif kernel_h % stride_h == 0:
        tile_h_pad_u = kernel_h - stride_h - ph_h
    else:
        tile_h_pad_u = kernel_h - kernel_h % stride_h - ph_h
    tile_h_pad_l = kernel_h - stride_h + ph_h
    tile_input_h = tile_h + tile_h_pad_u + tile_h_pad_l
    tile_h_out = (input_h - 1) // tile_h + 1

    if ph_h % stride_h == 0:
        pad_output_h = ph_h // stride_h
    else:
        pad_output_h = ph_h // stride_h + 1

    if tile_h_pad_u % stride_h == 0:
        pad_output_h -= tile_h_pad_u // stride_h
    else:
        pad_output_h -= tile_h_pad_u // stride_h + 1

    tile_output_h = (tile_input_h - kernel_h) // stride_h + 1

    tile_w = stride_w * tile_scale_w
    if kernel_w == stride_w:  # non-overlapping case
        tile_w_pad_u = pw_h % stride_w
    elif kernel_w % stride_w == 0:
        tile_w_pad_u = kernel_w - stride_w - pw_h
    else:
        tile_w_pad_u = kernel_w - kernel_w % stride_w - pw_h
    tile_w_pad_l = kernel_w - stride_w + pw_h
    tile_input_w = tile_w + tile_w_pad_u + tile_w_pad_l
    tile_w_out = (input_w - 1) // tile_w + 1

    if pw_h % stride_w == 0:
        pad_output_w = pw_h // stride_w
    else:
        pad_output_w = pw_h // stride_w + 1

    if tile_w_pad_u % stride_w == 0:
        pad_output_w -= tile_w_pad_u // stride_w
    else:
        pad_output_w -= tile_w_pad_u // stride_w + 1

    tile_output_w = (tile_input_w - kernel_w) // stride_w + 1

    def custom_maxpool_fdiff(out, inputs, head_, ad_attrs, new_pld_array):
        head_reshaped = akg.tvm.compute((batch_size, input_c1, tile_h_out, tile_w_out,
                                         tile_output_h, tile_output_w, input_c0),
                                        lambda b, c1, h_out, w_out, oh, ow, c0:
                                        akg.tvm.expr.Select(
                                            akg.tvm.any(h_out * tile_scale_h + pad_output_h + oh < 0,
                                                        h_out * tile_scale_h + pad_output_h + oh > out_size_h - 1,
                                                        w_out * tile_scale_w + pad_output_w + ow < 0,
                                                        w_out * tile_scale_w + pad_output_w + ow > out_size_w - 1),
                                            akg.tvm.const(0.0, dtype=dtype),
                                            head_(b, c1,
                                                  h_out * tile_scale_h + pad_output_h + oh,
                                                  w_out * tile_scale_w + pad_output_w + ow,
                                                  c0)),
                                        name="head_reshaped")

        mask_reshaped = akg.tvm.compute((batch_size, input_c1, tile_h_out, tile_w_out,
                                         tile_output_h, tile_output_w, kernel_h, kernel_w, input_c0),
                                        lambda b, c1, h_out, w_out, oh, ow, kh, kw, c0:
                                        akg.tvm.expr.Select(
                                            akg.tvm.any(h_out * tile_scale_h + pad_output_h + oh < 0,
                                                        h_out * tile_scale_h + pad_output_h + oh > out_size_h - 1,
                                                        w_out * tile_scale_w + pad_output_w + ow < 0,
                                                        w_out * tile_scale_w + pad_output_w + ow > out_size_w - 1),
                                            akg.tvm.const(0.0, dtype=dtype),
                                            mask(b, c1, kh, kw,
                                                 h_out * tile_scale_h + pad_output_h + oh,
                                                 w_out * tile_scale_w + pad_output_w + ow,
                                                 c0)),
                                        name="mask_reshaped")

        d_data = akg.tvm.compute((batch_size, input_c1, tile_h_out, tile_w_out,
                                  tile_output_h, tile_output_w, kernel_h, kernel_w, input_c0),
                                 lambda b, c1, h_out, w_out, oh, ow, kh, kw, c0:
                                 mask_reshaped(b, c1, h_out, w_out,
                                               oh, ow, kh, kw, c0)
                                 * head_reshaped(b, c1, h_out,
                                                 w_out, oh, ow, c0),
                                 name="d_data")

        data_reorg = akg.tvm.compute((batch_size, input_c1, tile_h_out, tile_w_out,
                                      tile_output_h, tile_output_w, tile_h, tile_w, input_c0),
                                     lambda b, c1, h_out, w_out, oh, ow, h, w, c0:
                                     akg.tvm.expr.Select(
                                         akg.tvm.any(h + tile_h_pad_u < oh * stride_h,
                                                     h + tile_h_pad_u > oh * stride_h + kernel_h - 1,
                                                     w + tile_w_pad_u < ow * stride_w,
                                                     w + tile_w_pad_u > ow * stride_w + kernel_w - 1),
                                         akg.tvm.const(0, dtype=dtype),
                                         d_data(b, c1, h_out, w_out, oh, ow,
                                                h + tile_h_pad_u - oh * stride_h,
                                                w + tile_w_pad_u - ow * stride_w,
                                                c0)),
                                     name="data_reorg")

        result_tile = akg.topi.sum(data_reorg, [4, 5])

        result = akg.tvm.compute(shape,
                                 lambda b, c1, h, w, c0:
                                 result_tile(b, c1, h // tile_h, w //
                                             tile_w, h % tile_h, w % tile_w, c0),
                                 name="result")
        return [result]

    # override differentiation computation with custom function
    [dl_ddata] = akg.differentiate(forward, [data], head, None, None,
                                   override={forward: ([data], custom_maxpool_fdiff)})

    # schedule for differetiation operation
    s = akg.tvm.create_schedule([dl_ddata.op])

    # get computations
    result = dl_ddata
    result_tile = result.op.input_tensors[0]
    data_reorg = result_tile.op.input_tensors[0]
    d_data = data_reorg.op.input_tensors[0]
    mask_reshaped = d_data.op.input_tensors[0]
    head_reshaped = d_data.op.input_tensors[1]

    def comp_func(s):

        data_ub = s.cache_read(mask, "local.UB", [mask_reshaped])
        head_ub = s.cache_read(head, "local.UB", [head_reshaped])
        result_ub = s.cache_write(result, "local.UB")

        s[d_data].set_scope("local.UB")
        s[data_reorg].set_scope("local.UB")
        s[mask_reshaped].set_scope("local.UB")
        s[head_reshaped].set_scope("local.UB")
        s[result_tile].set_scope("local.UB")

        s[result_ub].compute_inline()

        # inline inputs
        s[head_ub].compute_inline()
        s[data_ub].compute_inline()

        # result_tile dependencies
        s[data_reorg].compute_inline()
        b, c1, h_out, w_out, h, w, c0 = result_tile.op.axis
        oh, ow = result_tile.op.reduce_axis
        s[result_tile].reorder(b, c1, h_out, w_out, h, w, oh, ow, c0)

        s[d_data].compute_at(s[result_tile], w_out)
        s[mask_reshaped].compute_at(s[result_tile], w_out)
        s[head_reshaped].compute_at(s[result_tile], w_out)

        # tile result
        b, c1, h, w, c0 = result.op.axis
        h_out, h_in = s[result].split(h, tile_h)
        w_out, w_in = s[result].split(w, tile_w)
        s[result].reorder(b, c1, h_out, w_out, h_in, w_in, c0)
        s[result_tile].compute_at(s[result], w_out)

    return dl_ddata, comp_func


@utils.check_input_type((list, tuple), (list, tuple), (list, tuple), (str, list, tuple),
                        str, (bool, type(None)), (dict, type(None)))
def maxpool_ad_manual_schedule_all_max(shape, kernel, stride, pad, dtype, polyhedral=True, attrs=None):
    """automatic differentiate of maxpool with manual schedule for all maximum."""
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_h, pad_w, _, _ = pad
    batch_size, input_c1, input_h, input_w, input_c0 = shape
    pad_shape = (batch_size, input_c1, input_h + 2 *
                 pad_h, input_w + 2 * pad_w, input_c0)
    out_size_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_size_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1
    out_shape = (batch_size, input_c1, out_size_h, out_size_w, input_c0)

    def custom_maxpool_fdiff(out, inputs, head_, ad_attrs, new_pld_array):
        in_data = inputs[0]

        data_separated_by_windows = (
            kernel_h, kernel_w, batch_size, input_c1, out_size_h, out_size_w, input_c0)

        pad_data = akg.tvm.compute(pad_shape,
                                   lambda b, c1, h, w, c0:
                                   akg.tvm.expr.Select(
                                       akg.tvm.all(h >= pad_h,
                                                   h < input_h + pad_h,
                                                   w >= pad_w,
                                                   w < input_w + pad_w),
                                       in_data(b, c1, h - pad_h,
                                               w - pad_w, c0),
                                       akg.tvm.const(0.0, dtype=dtype)),
                                   name="pad_data")

        data_reshaped = akg.tvm.compute(data_separated_by_windows,
                                        lambda wh, ww, b, c1, oh, ow, c0:
                                        pad_data(b, c1, oh * stride_h +
                                                 wh, ow * stride_w + ww, c0),
                                        name="data_reshaped")

        max_broadcast = akg.tvm.compute(data_separated_by_windows,
                                        lambda wh, ww, b, c1, oh, ow, c0:
                                        out(b, c1, oh, ow, c0),
                                        name="max_broadcast")

        equal = akg.tvm.compute(data_separated_by_windows,
                                lambda wh, ww, b, c1, oh, ow, c0:
                                akg.tvm.expr.Select(
                                    max_broadcast(wh, ww, b, c1, oh, ow, c0) ==
                                    data_reshaped(wh, ww, b, c1, oh, ow, c0),
                                    head_(b, c1, oh, ow, c0),
                                    akg.tvm.const(0.0, dtype=dtype)),
                                name="equal")

        data_reorg = akg.tvm.compute((out_size_h, out_size_w, batch_size, input_c1, input_h + 2 * pad_h,
                                      input_w + 2 * pad_w, input_c0),
                                     lambda oh, ow, b, c1, h, w, c0:
                                     akg.tvm.expr.Select(
                                         akg.tvm.any(h < oh * stride_h,
                                                     h > oh * stride_h + kernel_h - 1,
                                                     w < ow * stride_w,
                                                     w > ow * stride_w + kernel_w - 1),
                                         akg.tvm.const(0, dtype=dtype),
                                         equal(h - oh * stride_h, w - ow * stride_w, b, c1, oh, ow, c0)),
                                     name="data_reorg")

        result_pad = akg.topi.sum(data_reorg, [0, 1])

        result = akg.tvm.compute(shape,
                                 lambda b, c1, h, w, c0:
                                 result_pad(b, c1, h + pad_h, w + pad_w, c0),
                                 name="result")

        return [result]

    # tensor for the input data
    data = akg.tvm.placeholder(shape, dtype, name="input_data")

    # maxpool output
    forward = akg.tvm.placeholder(out_shape, name="forward", dtype=dtype)

    # adjoint tensor for the differentiation
    head = akg.tvm.placeholder(out_shape, name="head", dtype=dtype)

    # override differentiation computation with custom function
    [dl_ddata] = akg.differentiate(forward, [data], head, None, None,
                                   override={forward: ([data], custom_maxpool_fdiff)})

    # schedule for differetiation operation
    s = akg.tvm.create_schedule([dl_ddata.op])

    # get computations
    result = dl_ddata
    result_pad = result.op.input_tensors[0]
    data_reorg = result_pad.op.input_tensors[0]
    equal = data_reorg.op.input_tensors[0]
    max_broadcast = equal.op.input_tensors[0]
    data_reshaped = equal.op.input_tensors[1]
    pad_data = data_reshaped.op.input_tensors[0]

    data_ub = s.cache_read(data, "local.UB", [pad_data])
    head_ub = s.cache_read(head, "local.UB", [equal])
    forward_ub = s.cache_read(forward, "local.UB", [max_broadcast])
    result_ub = s.cache_write(result, "local.UB")

    s[max_broadcast].set_scope("local.UB")
    s[data_reshaped].set_scope("local.UB")
    s[pad_data].set_scope("local.UB")
    s[equal].set_scope("local.UB")
    s[data_reorg].set_scope("local.UB")
    s[result_pad].set_scope("local.UB")

    s[data_ub].compute_inline()
    s[result_ub].compute_inline()
    s[pad_data].compute_inline()

    # equal dependencies
    s[forward_ub].compute_at(s[equal], equal.op.axis[0])
    s[max_broadcast].compute_at(s[equal], equal.op.axis[0])
    s[data_reshaped].compute_at(s[equal], equal.op.axis[0])
    s[head_ub].compute_at(s[equal], equal.op.axis[0])

    s[equal].compute_at(s[result_pad], result_pad.op.axis[0])

    # result dependencies
    s[data_reorg].compute_inline()
    b, c1, h, w, c0 = result_pad.op.axis
    oh, ow = result_pad.op.reduce_axis
    s[result_pad].reorder(oh, ow, b, c1, h, w, c0)

    b, c1, h, w, c0 = result.op.axis
    h_out, _ = s[result].split(h, stride_h)
    s[result_pad].compute_at(s[result], h_out)

    with akg.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [head, data, forward, dl_ddata], "cce", name="maxpool_ad_manual_schedule_all_max",
                        attrs=attrs, polyhedral=polyhedral)
        source_code = mod.imported_modules[0].get_source()
        kernel_name = "maxpool_ad_manual_schedule_all_max"
        create_code(kernel_name, './', source_code)
    return mod


def maxpool_ad_manual_schedule_no_overlap_all_max(shape, kernel, stride, pad, dtype, attrs=None, polyhedral=False):
    """automatic differentiate of maxpool with manual schedule for no overlap case."""
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_h, pad_w, _, _ = pad
    batch_size, input_c1, input_h, input_w, input_c0 = shape
    pad_shape = (batch_size, input_c1, input_h + 2 *
                 pad_h, input_w + 2 * pad_w, input_c0)

    def custom_maxpool_fdiff(out, inputs, head_, ad_attrs, new_pld_array):
        in_data = inputs[0]

        if stride_w != kernel_w:
            raise RuntimeError(
                "Only supports kernels with same dimensions as stride size!")
        if stride_h != kernel_h:
            raise RuntimeError(
                "Only supports kernels with same dimensions as stride size!")

        out_broadcast = akg.tvm.compute(pad_shape,
                                        lambda b, c1, h, w, c0:
                                        out(b, c1, akg.tvm.floordiv(
                                            h, stride_h), akg.tvm.floordiv(w, stride_w), c0),
                                        name="out_broadcast")

        # copy output to the shape of the padded input, copying the same value for the entire kernel size
        out_broadcast = akg.tvm.compute(pad_shape,
                                        lambda b, c1, h, w, c0:
                                        out(b, c1, akg.tvm.floordiv(
                                            h, stride_h), akg.tvm.floordiv(w, stride_w), c0),
                                        name="out_broadcast")

        # copy head to the shape of the padded input, copying the same value for the entire kernel size
        head_broadcast = akg.tvm.compute(pad_shape,
                                         lambda b, c1, h, w, c0:
                                         head_(b, c1, akg.tvm.floordiv(
                                             h, stride_h), akg.tvm.floordiv(w, stride_w), c0),
                                         name="head_broadcast")

        # check if value was a maximum and assign head of that position if it was
        # this is done for all the maximum values within one kernel
        result = akg.tvm.compute(in_data.shape,
                                 lambda b, c1, h, w, c0:
                                 akg.tvm.expr.Select(
                                     in_data(b, c1, h, w, c0) == out_broadcast(
                                         b, c1, h + pad_h, w + pad_w, c0),
                                     head_broadcast(
                                         b, c1, h + pad_h, w + pad_w, c0),
                                     akg.tvm.const(0, dtype=in_data.dtype)),
                                 name="result")
        return [result]

    out_size_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_size_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1

    out_shape = (batch_size, input_c1, out_size_h, out_size_w, input_c0)

    # tensor for the input data
    data = akg.tvm.placeholder(shape, dtype, name="input_data")

    # maxpool output
    forward = akg.tvm.placeholder(out_shape, name="forward", dtype=dtype)

    # adjoint tensor for the differentiation
    head = akg.tvm.placeholder(out_shape, name="head", dtype=dtype)

    # override differentiation computation with custom function
    [dl_ddata] = akg.differentiate(forward, [data], head, None, None,
                                   override={forward: ([data], custom_maxpool_fdiff)})

    # schedule for differetiation operation
    s = akg.tvm.create_schedule([dl_ddata.op])

    # get computations
    result = dl_ddata
    forward_broadcast = result.op.input_tensors[1]
    head_broadcast = result.op.input_tensors[2]

    # cache reads and writes
    result_ub = s.cache_write(result, "local.UB")
    data_ub = s.cache_read(data, "local.UB", [result_ub])
    head_ub = s.cache_read(head, "local.UB", [head_broadcast])
    forward_ub = s.cache_read(forward, "local.UB", [forward_broadcast])

    s[head_broadcast].set_scope("local.UB")
    s[forward_broadcast].set_scope("local.UB")

    s[head_ub].compute_at(s[head_broadcast], head_broadcast.op.axis[0])
    s[forward_ub].compute_at(s[forward_broadcast],
                             forward_broadcast.op.axis[0])
    s[data_ub].compute_at(s[result_ub], result_ub.op.axis[0])
    s[forward_broadcast].compute_at(s[result_ub], result_ub.op.axis[0])
    s[head_broadcast].compute_at(s[result_ub], result_ub.op.axis[0])

    _, c1, h, _, _ = result.op.axis

    if input_h + 2 * pad_h > 32 or input_w + 2 * pad_w > 32:
        h_outer, _ = s[result].split(h, 4)
        s[result_ub].compute_at(s[result], h_outer)
    else:
        s[result_ub].compute_at(s[result], c1)

    with akg.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [head, data, forward, dl_ddata], "cce",
                        name="maxpool_ad_manual_schedule_no_overlap_all_max", attrs=attrs, polyhedral=polyhedral)
        source_code = mod.imported_modules[0].get_source()
        kernel_name = "maxpool_ad_manual_schedule_no_overlap_all_max"
        create_code(kernel_name, './', source_code)
    return mod
