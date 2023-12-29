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

"""operator dsl function: avgpool"""
import akg.tvm
import akg.utils as utils
from akg.tvm.hybrid import script
from akg import dim
from akg.dim import DIM
from akg.utils import custom_tiling as ct_util
from akg.utils.dsl_create import cal_pad_shapes_by_strategy, zero_const
from akg.utils.format_transform import get_shape
from akg.ops.nn.ascend.maxpool import img2col


def avgpool_set_dim_func(a_value, kernel, stride, pad):
    """set dim info to attr with avgpool_set_dim_map"""
    avgpool_set_dim_map = {
        str(((1, 1, 16, 16, 16), (4, 4), (3, 3), 'VALID', 'float16')): ((16, 1), (20, 1), (5, 1)),
        str(((1, 1, 16, 16, 16), (4, 4), (3, 3), (0, 0, 0, 0), 'float16')): ((16, 1), (20, 1), (5, 1)),
        str(((10, 3, 16, 16, 16), (4, 4), (3, 3), (0, 0, 0, 0), 'float16')): ((2, 2), (3, 3), (16, 16), (5, 5), (5, 5)),
        str(((1, 2, 16, 16, 16), (4, 4), (3, 3), (1, 1, 1, 1), 'float16')): ((1, 1), (16, 16), (19, 19)),
    }
    key = []
    key.append(tuple(get_shape(a_value)))
    key.append(kernel)
    key.append(stride)
    if isinstance(pad, list):
        pad = tuple(pad)
    key.append(pad)
    key.append(a_value.dtype)
    hash_key = str(tuple(key))

    if hash_key in avgpool_set_dim_map.keys():
        return ct_util.set_dims(avgpool_set_dim_map[hash_key]), hash_key
    return "", hash_key


def avg_pool_5d_hybrid(a_value, kernel, stride, strategy):
    """avgpool with 5d case via hybrid"""
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    shape = get_shape(a_value)
    batch_size, c1_, in_size_h, in_size_w, c0_ = shape
    dtype = a_value.dtype
    if len(shape) != 5:
        raise ValueError("Only support 5-dim pooling!")
    if len(kernel) != 2:
        raise ValueError("Only support 2-dim kernel!")

    [pad_height_head, _, pad_width_head, _], [out_size_h, out_size_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, strategy)

    avg_pre = akg.tvm.const(1.0000 / (kernel_w * kernel_h), dtype=dtype)
    zero = akg.tvm.const(0.0, dtype=dtype)

    @script(capture=locals())
    def avg_pool_hybrid(inputs, zero, avg_pre):
        output = output_tensor(
            (batch_size, c1_, out_size_h, out_size_w, c0_), inputs.dtype)

        for n in range(batch_size):
            for c1 in range(c1_):
                # Head
                for ow in range(out_size_w):
                    for c0 in range(c0_):
                        output[n, c1, 0, ow, c0] = zero
                for ow in range(out_size_w):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            for c0 in range(c0_):
                                if (kh >= pad_height_head) \
                                        and (ow * stride_w + kw - pad_width_head >= 0) \
                                        and (ow * stride_w + kw <= in_size_w + pad_width_head - 1):
                                    output[n, c1, 0, ow, c0] = output[n, c1, 0, ow, c0] +\
                                        inputs[n, c1, kh - pad_height_head,
                                               ow * stride_w + kw - pad_width_head, c0]
                                else:
                                    output[n, c1, 0, ow, c0] += zero
                for ow in range(out_size_w):
                    for c0 in range(c0_):
                        output[n, c1, 0, ow, c0] *= avg_pre
                # Tail
                for oh in range(out_size_h - 1):
                    for ow in range(out_size_w):
                        for c0 in range(c0_):
                            output[n, c1, oh + 1, ow, c0] = zero
                for oh in range(out_size_h - 1):
                    for ow in range(out_size_w):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                for c0 in range(c0_):
                                    if ((oh + 1) * stride_h + kh <= in_size_h + pad_height_head - 1)\
                                            and (ow * stride_w + kw >= pad_width_head)\
                                            and (ow * stride_w + kw <= in_size_w + pad_width_head - 1):
                                        output[n, c1, oh + 1, ow, c0] = output[n, c1, oh + 1, ow, c0] +\
                                            inputs[n, c1, (oh + 1) * stride_h +
                                                   kh - pad_height_head, ow * stride_w +
                                                   kw - pad_width_head, c0]
                                    else:
                                        output[n, c1, oh + 1, ow, c0] += zero
                for oh in range(out_size_h - 1):
                    for ow in range(out_size_w):
                        for c0 in range(c0_):
                            output[n, c1, oh + 1, ow, c0] *= avg_pre
        return output

    res_value = avg_pool_hybrid(a_value, zero, avg_pre)

    # set dim
    info = dim.Dim()
    # first part
    info.setdim(index=0, axis=0, tilel1=out_size_w, tilel0=0)  # ow
    info.setdim(index=0, axis=1, tilel1=c0_, tilel0=0)  # c0
    info.setdim(index=0, axis=2, tilel1=kernel_h, tilel0=0)  # kh

    # second part
    info.setdim(index=1, axis=0, tilel1=out_size_h - 1, tilel0=0)  # oh-1
    info.setdim(index=1, axis=1, tilel1=out_size_w, tilel0=0)  # ow
    info.setdim(index=1, axis=2, tilel1=c0_, tilel0=0)  # c0
    info.setdim(index=1, axis=3, tilel1=kernel_h, tilel0=0)  # kh

    info = str(info)

    attrs = {DIM: info}
    return res_value, attrs


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple),
                        (list, tuple), (str, list, tuple), (str, type(None)))
def avgpool(data, kernel, stride, strategy, target=utils.CCE):
    """
    Performs the average pooling on the input datas.

    Note:
        Only support 5D format(NC1HWC0), and pooling will work on H and W.
        Support **Strategies**:

        .. hlist::
          * VALID: will not pad, and drop tailed elements when pooling.
                   Output shape will be  `ceil((pool_shapes[i] - (kernel[i] - 1)) / stride[i])`
            > **example**:
            > params: inputs => 11, kernel width => 5, stride => 4
            > inputs: 1  2  3  4  5  6  7  8  9  10 11
            > 1st window contains: 1 2 3 4 5
            > 2nd window contains: 5 6 7 8 9
            > dropped: 10 11
          * SAME: will pad with zero evenly each side, but will add extra to tail
                  if the total padding amount is odd.
                  Output shape will be  `ceil(pool_shapes[i] / stride[i])`
            > **example**:
            > params: inputs => 10, kernel width => 5, stride => 4
            > inputs: 1  2  3  4  5  6  7  8  9  10
            > paded: 0(pad1) | 1  2  3  4  5  6  7  8  9  10 | 0(pad2) 0(pad3)
            > 1st window contains: 0(pad1) 1 2 3 4
            > 2nd window contains: 4 5 6 7 8
            > 3rd window contains: 8 9 10 0(pad2) 0(pad3)
            > dropped: None
          * CONSTANTS: will pad with zero according to given constants
                       (also dropped tailed elements when pooling).
            > **example**:
            > params: inputs => 10, kernel width => 5, stride => 4, pad => (2, 2)
            > inputs: 1  2  3  4  5  6  7  8  9  10
            > paded: 0(pad1) 0(pad2) | 1  2  3  4  5  6  7  8  9  10 | 0(pad2) 0(pad3)
            > 1st window contains: 0(pad1) 0(pad2) 1 2 3
            > 2nd window contains: 3 4 5 6 7
            > 3rd window contains: 7 8 9 10 0(pad3)
            > dropped: 0(pad4)

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.
        kernel (Union[list, tuple]): List or tuple of two int numbers for pooling window's size.
        stride (Union[list, tuple]): List or tuple of two int numbers for window's stride.
        strategy (Union[str, list, tuple]): A string or list or tuple for padding strategy,
            should be 'VALID', 'SAME' or instance of list(including four int numbers,
            as 'CONSTANTS' strategy).

    Returns:
        Tensor as result for average pooling.

    Supported Platforms:
        'Ascend'
    """
    dim_info, _ = avgpool_set_dim_func(data, kernel, stride, strategy)
    attrs = {DIM: dim_info}
    attrs['disable_half_to_float_sum_opt'] = True
    attrs['pragma_disable_whole_component'] = False

    shape = [x.value for x in data.shape]
    dtype = data.dtype
    utils.davinci_format_check(shape, "NC1HWC0", dim=5)
    utils.check_shape(kernel, 2, 'Kernel')
    utils.check_shape(stride, 2, 'Stride')

    if shape[2] > 60 and shape[3] > 60:
        return avg_pool_5d_hybrid(data, kernel, stride, strategy)

    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    batch_size, c1, in_size_h, in_size_w, c0 = shape

    [pad_height_head, pad_height_tail, pad_width_head, pad_width_tail], [out_size_h, out_size_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, strategy)

    pad_shape = (batch_size,
                 c1,
                 in_size_h + pad_height_head + pad_height_tail,
                 in_size_w + pad_width_head + pad_width_tail,
                 c0)

    pad2d = akg.tvm.compute(pad_shape,
                            lambda n, c1, h, w, c0:
                            akg.tvm.if_then_else(
                                akg.tvm.any(
                                    h < pad_height_head,
                                    h > in_size_h + pad_height_head - 1,
                                    w < pad_width_head,
                                    w > in_size_w + pad_width_head - 1
                                ),
                                akg.tvm.const(0.0, dtype=dtype),
                                data[n, c1, h - pad_height_head, w - pad_width_head, c0],
                            ),
                            name="pad2d")

    axis_kernel_h = akg.tvm.reduce_axis((0, kernel_h), name="axis_kernel_h")
    axis_kernel_w = akg.tvm.reduce_axis((0, kernel_w), name="axis_kernel_w")

    out_shape = (batch_size, c1, out_size_h, out_size_w, c0)

    dividor = akg.tvm.const(kernel_h * kernel_w, dtype)
    res = akg.tvm.compute(out_shape,
                          lambda n, c1, h, w, c0:
                          akg.tvm.sum(
                              pad2d[n, c1, h * stride_h + axis_kernel_h, w * stride_w + axis_kernel_w, c0],
                              axis=[axis_kernel_h, axis_kernel_w]
                          ),
                          name="res")
    res_value = akg.tvm.compute(out_shape,
                                lambda n, c1, h, w, c0:
                                res[n, c1, h, w, c0] / dividor,
                                name="res_value")
    return res_value, attrs


def avgpool_with_img2col(data, kernel, stride, strategy):
    """
    Performs the avgpool with img2col.

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
        tvm.tensor.Tensor, result for gradient of avgpooling.
    """
    shape = get_shape(data)
    dtype = data.dtype

    utils.davinci_format_check(shape, "NC1HWC0", dim=5)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.FLOAT16)
    utils.check_shape(kernel, 2, "Kernel")
    utils.check_shape(stride, 2, "Stride")

    kernel_h, kernel_w = kernel
    in_n, in_c1, _, _, in_c0 = shape

    [ph_h, ph_t, pw_h, pw_t], [out_h, out_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, strategy)

    pad = [ph_h, ph_t, pw_h, pw_t]
    pad_value = zero_const(dtype)

    # fmap img2col l1 -> ub in zZ format by fractal
    fmap_img2col_shp_ub = (in_n, in_c1, kernel_h, kernel_w, out_h, out_w, in_c0)
    fmap_img2col_ub = img2col(data, fmap_img2col_shp_ub, kernel_h, kernel_w,
                              pad, stride, pad_value, tag="")

    out_shape = (in_n, in_c1, out_h, out_w, in_c0)
    reduce_axis_h = akg.tvm.reduce_axis((0, kernel_h), name="reduce_h")
    reduce_axis_w = akg.tvm.reduce_axis((0, kernel_w), name="reduce_w")
    res_sum = akg.tvm.compute(out_shape,
                              lambda n, c1, oh, ow, c0:
                              akg.tvm.sum(
                                  fmap_img2col_ub[n, c1, reduce_axis_h,
                                                  reduce_axis_w, oh, ow, c0],
                                  axis=[reduce_axis_h, reduce_axis_w]),
                              name="pooling_avg")

    dividor = akg.tvm.const(kernel_h * kernel_w, dtype)
    output = akg.tvm.compute(out_shape, lambda *i: res_sum(*i) / dividor,
                             name="res_value")
    return output
