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

"""operator dsl function: conv_input_ad"""
import akg.tvm
import akg.topi
import akg
from akg.ops.nn.ascend.conv_backprop_input import conv_backprop_input
from akg.ops.nn.ascend.conv import Conv
from akg.utils.format_transform import tvm_array_to_list
import akg.utils as utils


def expr_to_int(in_expr):
    """Converte expr to int type value."""
    result = [a.value for a in in_expr]
    return result


@akg.tvm.register_func("akg.autodiff.conv_input_ad_tensor")
def conv_input_ad_tensor(data, fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=None):
    """wraper of convolution filter backprop func."""
    data_list = tvm_array_to_list(data)
    fmap_shape = expr_to_int(fmap_shape)
    filter_shape = expr_to_int(filter_shape)
    pad_ = expr_to_int(pad_)
    stride_ = expr_to_int(stride_)
    dilation_ = expr_to_int(dilation_)
    c, _ = conv_backprop_input(data_list, fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=attrs)
    return c


def conv_input_ad_config(data, fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=None):
    """Configuration of convolution filter gradient."""
    _, configs = conv_backprop_input(data, fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=attrs)
    return configs


@utils.check_input_type((list, tuple), (list, tuple), (list, tuple),
                          (list, tuple), (list, tuple), (list, tuple), (dict, type(None)), (str, type(None)))
def ConvInputAd(input_ad_inputs, fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=None, target=utils.CCE):
    """
    Compute dx according to "conv forward".

    Args:
        input_ad_inputs (list[tvm.tensor.Tensor]): a list with length 2.
              input_ad_inputs[0](consider as dy) Tensor of type float16 ,shape 5D(out_n, out_c//C0, out_h, out_w,C0)
              input_ad_inputs[1](consider as w)  Tensor of type float16 ,shape 4D(wC//C0*wH*wW, wN//C0, C0,C0)
        fmap_shape (list): [fN, fC, fH, fW]
        filter_shape (list): [wN, wC, wH, wW]
        pad_ (list): [pad_left, pad_right, pad_top, pad_bottom]
        stride_ (list): [stride_h, stride_w]
        dilation_ (list): [dilation_h, dilation_w]
        attrs (dict): a dict with keys like conv_tile, bypass and etc.

    Returns:
        tvm.tensor.Tensor, configs.
    
    Supported Platforms:
        'Ascend'
    """

    backward_dy, forward_w = input_ad_inputs

    in_n, in_c, in_h, in_w = fmap_shape
    block_size = 16
    in_c = (in_c + block_size - 1) // block_size * block_size
    x_5d_shape = (in_n, in_c // block_size, in_h, in_w, block_size)

    forward_x = akg.tvm.placeholder(x_5d_shape, forward_w.dtype, "input_X")
    original_filter_shape = akg.tvm.placeholder(filter_shape, forward_w.dtype, "input_filter")
    forward_output, _ = Conv([forward_x, forward_w], fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias=False, attrs=attrs)

    ad_attrs = {"ad_conv_enable": 1, "ad_conv_reuse_conv": 0}
    jacs = list(akg.differentiate(forward_output, [forward_x], backward_dy, ad_attrs,
                                  [backward_dy, forward_w, original_filter_shape]))
    configs = conv_input_ad_config([backward_dy, forward_w], fmap_shape, filter_shape,
                                   pad_, stride_, dilation_, attrs=attrs)

    return jacs[0], configs
