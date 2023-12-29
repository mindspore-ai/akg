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

"""operator dsl function: maxpool_grad"""
import akg.tvm
from akg.tvm.hybrid import script
import akg.utils as utils
from akg.utils import custom_tiling as ct_util
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from akg.utils.kernel_exec import product_is_mini
from akg.dim import DIM
from akg.utils.format_transform import get_shape


def get_attrs():
    """get attrs config"""
    attr_map = {
        "disable_cse": 1,
        "pragma_disable_schedule_shift": 1,
        "pragma_opt_for_dsa": 1,
        "loop_partition_unroll": False,
        "enable_pre_poly_loop_partition": False
    }
    return attr_map


set_attr_map_ = {
    str(((2, 16, 40, 24, 16), (1, 1), (2, 2), (0, 0, 0, 0))): (
        ("pragma_disable_whole_component", 0),),
}

maxpool_grad_dim_map = {
    str(((2, 16, 40, 24, 16), (1, 1), (2, 2), (0, 0, 0, 0))): (
        (1, 1), (2, 1), (20, 1), (16, 1)),
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME')): (
        (1, 1), (1, 1), (16, 1), (1, 1)),
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1))): (
        (1, 1), (1, 1), (16, 1), (1, 1)),
    str(((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 0, 1, 0))): (
        (1, 1), (1, 1), (16, 1), (1, 1)),
    str(((1, 1, 4, 4, 16), (2, 2), (2, 2), (0, 0, 0, 0))): (
        (0, 1), (0, 1), (0, 1)),
    str(((1, 1, 16, 16, 16), (4, 4), (4, 4), (0, 0, 0, 0))): (
        (0, 1), (0, 1), (0, 1)),
    str(((1, 1, 32, 32, 16), (4, 4), (4, 4), (0, 0, 0, 0))): (
        (0, 1), (0, 1), (0, 1)),
    str(((32, 16, 28, 28, 16), (3, 3), (2, 2), 'VALID')): (
        (8, 1), (1, 1), (16, 1), (1, 1)),
    str(((32, 16, 13, 13, 16), (3, 3), (2, 2), 'VALID')): (
        (8, 1), (1, 1), (16, 1), (1, 1)),
    str(((32, 6, 57, 57, 16), (3, 3), (2, 2), 'VALID')): (
        (4, 1), (1, 1), (16, 1), (1, 1)),
}


def maxpool_grad_set_dim_func(x, y, dy, kernel, stride, pad):
    """dim func for maxpool grad"""
    key = str((tuple(x.shape), tuple(kernel), tuple(stride), pad))
    attrs = {}
    if key in set_attr_map_.keys():
        for attr in set_attr_map_[key]:
            attrs[attr[0]] = attr[1]
    if key in maxpool_grad_dim_map.keys():
        return ct_util.set_dims_by_key(key, maxpool_grad_dim_map), key, attrs
    return "", key, attrs


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, (list, tuple), (list, tuple),
                          (str, list, tuple), (str, type(None)))
def MaxpoolGrad(x, y, dy, kernel, stride, pad, target=utils.CCE):
    """
    Performs the gradient of maxpool pooling on the input datas.

    Note:
        Only support 5D format(NC1HWC0), and pooling will work on H and W.

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.
        y (tvm.tensor.Tensor): Tensor, the maxpool result.
        dy (tvm.tensor.Tensor): Tensor, the gradient needed to be propagation.
        kernel (Union[List, Tuple]): two int numbers for pooling window's size.
        stride (Union[List, Tuple]): two int numbers for window's stride.
        pad (Union[String, List, Tuple]): padding, should be 'VALID','SAME' or
            instance of list(four int numbers, as 'CONSTANTS' strategy).
            Support **pad** is the same as avgpool's **Strategies**.

    Returns:
        Tensor as result for gradient of maxpooling.
    
    Supported Platforms:
        'Ascend'
    """
    attrs = get_attrs()
    dim_info, _, attrs_info = maxpool_grad_set_dim_func(x, y, dy, kernel, stride, pad)
    attrs.update(attrs_info)
    attrs[DIM] = dim_info

    shape = get_shape(x)
    ori_dtype = x.dtype
    utils.ops_dtype_check(ori_dtype, utils.DtypeForDavinci.ALL_FLOAT)

    if product_is_mini() and ori_dtype == 'float32':
        raise RuntimeError("Maxpool only support"
                           "\'float16\' while platform is mini_v100!")
    dtype = ori_dtype

    if len(shape) != 5:
        raise ValueError("Only support 5-dim pooling!")
    if shape[-1] % 16 != 0:
        raise ValueError("Last shape must be divisible by 16!")
    if len(kernel) != 2:
        raise ValueError("Only support 2-dim kernel!")
    if len(stride) != 2:
        raise ValueError("Only support 2-dim stride!")
    if not isinstance(pad, str) \
            and not (isinstance(pad, (list, tuple)) and len(pad) == 4):
        raise ValueError("Only support string or list/tuple of 4 int numbers!")

    utils.check_shape(shape)

    in_n, in_c1, in_h, in_w, in_c0 = shape
    k_h, k_w = kernel
    s_h, s_w = stride
    [ph_h, ph_t, pw_h, pw_t], [y_h, y_w] = \
        cal_pad_shapes_by_strategy(shape, kernel, stride, pad)
    k_h_hybrid = k_h
    k_w_hybrid = k_w

    yn = in_n
    yc1 = in_c1
    yc0 = in_c0

    @script(capture=locals())
    def max_pool_grad_hybrid(zero_, one_, min_value_, x_, y_, dy_):
        x_dummy_ = allocate((in_n, in_c1, ph_h + in_h + ph_t, pw_h + in_w + pw_t, in_c0),
                            x_.dtype, "local")
        x_img_ = allocate((yn, yc1, y_h, y_w, k_h_hybrid, k_w_hybrid, yc0),
                          x_.dtype, "local")
        y_img_ = allocate((yn, yc1, y_h, y_w, k_h_hybrid, k_w_hybrid, yc0),
                          x_.dtype)
        mask_ = allocate((yn, yc1, y_h, y_w, k_h_hybrid, k_w_hybrid, yc0),
                         x_.dtype)
        mask_new = allocate((yn, yc1, y_h, y_w, k_h_hybrid, k_w_hybrid, yc0),
                            dy_.dtype)
        mask_res = allocate((yn, yc1, y_h, y_w, k_h_hybrid, k_w_hybrid, yc0),
                            dy_.dtype)
        output_pre = allocate((yn, yc1, y_h, y_w, k_h_hybrid, k_w_hybrid, yc0),
                              dy_.dtype)
        output_dummy_body = allocate((in_n, in_c1,
                                      ph_h + in_h + ph_t, pw_h + in_w + pw_t, in_c0), dy_.dtype)
        output = output_tensor((in_n, in_c1, in_h, in_w, in_c0), dy_.dtype)

        for n in range(yn):
            for c1 in range(yc1):
                for h in range(y_h):

                    for kh in range(k_h_hybrid):
                        for iw in range(pw_h + in_w + pw_t):
                            for c0 in range(yc0):
                                x_dummy_[n, c1,
                                         h * s_h + kh, iw, c0] = min_value_
                                output_dummy_body[n, c1,
                                                  h * s_h + kh, iw, c0] = zero_

                    for kh in range(k_h_hybrid):
                        for iw in range(in_w):
                            for c0 in range(yc0):
                                if (h * s_h + kh >= ph_h
                                        and h * s_h + kh < in_h + ph_h):
                                    x_dummy_[n, c1, h * s_h + kh,
                                             iw + pw_h, c0] = \
                                        x_[n, c1, h * s_h + kh - ph_h, iw, c0]

                    for kh in range(k_h_hybrid):
                        for iw in range(in_w):
                            for c0 in range(yc0):
                                if (h * s_h + kh >= ph_h
                                        and h * s_h + kh < in_h + ph_h):
                                    output_dummy_body[n, c1,
                                                      h * s_h + kh, iw + pw_h, c0] = \
                                        output[n, c1, h * s_h + kh - ph_h, iw, c0]

                    for w in range(y_w):
                        for kh in range(k_h_hybrid):
                            for kw in range(k_w_hybrid):
                                for c0 in range(yc0):
                                    x_img_[n, c1, h, w, kh, kw, c0] = \
                                        x_dummy_[n, c1, h * s_h + kh,
                                                 w * s_w + kw, c0]
                                    y_img_[n, c1, h, w, kh, kw, c0] = \
                                        y_[n, c1, h, w, c0]
                                    mask_[n, c1, h, w, kh, kw, c0] = zero_ \
                                        if x_img_[n, c1, h, w, kh, kw, c0] \
                                        < y_img_[n, c1, h, w, kh, kw, c0] \
                                        else one_
                        for kh in range(k_h_hybrid):
                            for kw in range(k_w_hybrid):
                                for c0 in range(yc0):
                                    mask_new[n, c1, h, w, kh,
                                             kw, c0] = zero_
                                for kh_0 in range(kh):
                                    for kw_0 in range(k_w_hybrid):
                                        for c0 in range(yc0):
                                            mask_new[n, c1, h, w, kh,
                                                     kw, c0] = \
                                                mask_new[n, c1, h, w,
                                                         kh, kw, c0] \
                                                + mask_[n, c1, h, w,
                                                        kh_0, kw_0, c0]
                                for kw_0 in range(kw + 1):
                                    for c0 in range(yc0):
                                        mask_new[n, c1, h, w, kh, kw, c0] = \
                                            mask_new[n, c1, h, w, kh, kw, c0] \
                                            + mask_[n, c1, h, w, kh, kw_0, c0]
                        for kh in range(k_h_hybrid):
                            for kw in range(k_w_hybrid):
                                for c0 in range(yc0):
                                    mask_res[n, c1, h, w, kh, kw, c0] = \
                                        zero_ \
                                        if mask_new[n, c1, h, w, kh, kw, c0] \
                                        > mask_[n, c1, h, w, kh, kw, c0] \
                                        else mask_[n, c1, h, w, kh, kw, c0]
                                    output_pre[n, c1, h, w, kh, kw, c0] = \
                                        mask_res[n, c1, h, w, kh, kw, c0] \
                                        * dy_[n, c1, h, w, c0]
                                    output_dummy_body[n, c1,
                                                      h * s_h + kh, w * s_w + kw, c0] += \
                                        output_pre[n, c1, h, w, kh, kw, c0]
                    for kh in range(k_h_hybrid):
                        for iw in range(in_w):
                            for c0 in range(yc0):
                                if (h * s_h + kh >= ph_h
                                        and h * s_h + kh < in_h + ph_h):
                                    output[n, c1, h * s_h + kh - ph_h,
                                           iw, c0] = \
                                        output_dummy_body[n, c1,
                                                          h * s_h + kh, iw + pw_h, c0]

        return output

    zero = akg.tvm.const(0.0, dtype=dtype)
    one = akg.tvm.const(1.0, dtype=dtype)
    min_value = akg.tvm.const(-65504.0 if dtype == 'float16'
                              else -340282346638528859811704183484516925440.0, dtype=dtype)
    output = max_pool_grad_hybrid(zero, one, min_value, x, y, dy)
    return output, attrs
