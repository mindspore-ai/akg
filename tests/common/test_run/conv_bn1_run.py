# Copyright 2019 Huawei Technologies Co., Ltd
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

""" conv_bn1 run function """

import os
import numpy as np
from akg.utils import kernel_exec as utils

from akg.ops.nn import conv_bn1
from gen_random import random_gaussian
from test_run.conv_utils import conv_param_prepare, conv_shape_4d, conv_forward_naive, conv_tensor_4d_to_5d
from akg.utils import validation_check as vc_util
from base import get_rtol_atol
from tensorio import compare_tensor


def gen_data(fm_shape, w_shape, pad, stride, dilation, bias):

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    stride, pad, dilation = conv_param_prepare(conv_param)
    fm_shape, w_shape, out_shape = conv_shape_4d(fm_shape, w_shape, pad, stride, dilation)
    IN, IC, IH, IW = fm_shape
    WN, WC, WH, WW = w_shape

    x = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    w = random_gaussian((WN, WC, WH, WW), miu=0.5, sigma=0.01).astype(np.float16)

    b = (np.array(np.zeros(WN))).astype(np.float16, copy=False)

    out = conv_forward_naive(x.astype(np.float32), w.astype(np.float32), b, conv_param)
    feature, filter, bb, output = conv_tensor_4d_to_5d(x, w, b, out)

    return feature, filter, bb, output

def conv_bn1_run(fmap_shape, filter_shape, pad, stride, dilation,
                 use_bias=False, attrs=None):
    vc_util.convolution_format_check(fmap_shape, filter_shape, pad, stride, dilation)
    if use_bias:
        raise ValueError("do not support bias yet !!!")

    conv_dtype = 'float16'
    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    stride, pad, dilation = conv_param_prepare(conv_param)
    fm_shape, w_shape, out_shape = conv_shape_4d(fmap_shape, filter_shape, pad, stride, dilation)
    IN, IC, IH, IW = fm_shape
    WN, WC, WH, WW = w_shape
    C0 = 16

    input_shape = [(IN, IC // C0, IH, IW, C0), (WC // C0 * WH * WW, WN // 16, 16, C0)]
    mod = utils.op_build_test(conv_bn1.conv_bn1, [input_shape], [conv_dtype],
                              op_attrs=[fmap_shape, filter_shape, pad, stride, dilation, use_bias, attrs],
                              kernel_name='conv_bn1', attrs=attrs)

    fmap_data, filter_data, bias_data, conv_expect = \
        gen_data(fmap_shape, filter_shape, pad, stride, dilation, use_bias)

    axes = (0, 2, 3)
    conv_mean = np.mean(conv_expect, axis=axes, keepdims=True)
    conv_square = np.power(conv_expect, 2)
    conv_var_part = np.mean(conv_square, axis=axes, keepdims=True)

    expects = (conv_expect, conv_var_part, conv_mean)

    out_datas = [np.full(e.shape, 0, 'float16') for e in expects]
    out_datas[1] = out_datas[1].astype(np.float32)
    out_datas[2] = out_datas[2].astype(np.float32)

    in_data = [fmap_data, filter_data]

    args = in_data
    for out in out_datas:
        args.append(out)
    args = tuple(args)

    outputs = utils.mod_launch(mod, args, outputs=(-3, -2, -1), expect=expects)
    rtol, atol = get_rtol_atol("conv_bn1", conv_dtype)
    cmp_res = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), outputs, expects))
    return (fmap_data, filter_data, bias_data), outputs, expects, all(cmp_res)
