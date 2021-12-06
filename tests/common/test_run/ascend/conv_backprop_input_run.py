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

import os
import numpy as np
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.ops.nn.ascend import ConvBackpropInput
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils.kernel_exec import gen_kernel_name
from tests.common.base import get_rtol_atol


def conv_backprop_input_run(fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=None):
    conv_dtype = 'float16'
    block_size = 16

    vc_util.convolution_format_check(fmap_shape, filter_shape, pad_, stride_, dilation_)

    in_n, in_c, in_h, in_w = fmap_shape
    cout, cin, w_h, w_w = filter_shape

    in_c = (in_c + block_size - 1) // block_size * block_size
    cout = (cout + block_size - 1) // block_size * block_size

    pad_top, pad_bottom, pad_left, pad_right = pad_
    stride_h, stride_w = stride_

    out_n = in_n
    out_c = cout
    out_h = (in_h + pad_top + pad_bottom - w_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - w_w) // stride_w + 1

    x_shape = (out_n, out_c, out_h, out_w)
    w_shape = (cout, in_c, w_h, w_w)
    inN, inC, inH, inW = x_shape
    input_shape_nc1hwc0 = (inN, inC // block_size, inH, inW, block_size)
    k_n, k_c, k_h, k_w = w_shape
    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0
    kernel_shape_fractal = (k_c // block_size * k_h * k_w, k_n // block_size, block_size, block_size)

    input_shape = [input_shape_nc1hwc0, kernel_shape_fractal]

    input_file = os.environ.get("RANDOM_DATA_DISK_PATH", "")
    expect_file = input_file + "/" + gen_kernel_name([input_shape], [conv_dtype],
                              op_attrs=[fmap_shape, filter_shape, pad_, stride_, dilation_, attrs],
                              kernel_name='conv_backprop_input', attrs=attrs) + ".bin"
    fmap_data, filter_data, expect = gen_data(fmap_shape, filter_shape, pad_, stride_, dilation_, expect_file, attrs=attrs)

    out_data = np.full(expect.shape, np.nan, 'float16')
    input = (fmap_data, filter_data)

    flag_w = os.environ.get("WRITE_TO_DISK", "No")
    if flag_w == "Yes":
        return input, out_data, expect, True

    mod = utils.op_build_test(ConvBackpropInput, [input_shape], [conv_dtype],
                              op_attrs=[fmap_shape, filter_shape, pad_, stride_, dilation_, attrs],
                              kernel_name='conv_backprop_input', attrs=attrs)

    args = (fmap_data, filter_data, out_data)
    out_data = utils.mod_launch(mod, args, expect=expect)
    rtol, atol = get_rtol_atol("conv_backprop_input", conv_dtype)
    return input, out_data, expect, compare_tensor(out_data, expect, rtol=rtol, atol=atol, equal_nan=True)


def conv_backprop_input_naive(x, w, dy, pad_list, stride_list):
    N, C, H, W = dy.shape
    Cin, Cout, KH, KW = w.shape
    assert(C == Cin)

    pad_top, pad_bottom, pad_left, pad_right = pad_list
    stride_h, stride_w = stride_list

    if stride_h > 1 or stride_w > 1:
        dy_ = np.full((N, C, H * stride_h, W * stride_w), 0, np.float32)
        for nn in range(N):
            for nc in range(C):
                for nh in range(H):
                    for nw in range(W):
                        dy_[nn, nc, nh * stride_h, nw * stride_w] = dy[nn, nc, nh, nw]
        dy = dy_
        H = H * stride_h
        W = W * stride_w
        stride_h = 1
        stride_w = 1

    H_out = (H + pad_top + pad_bottom - KH) // stride_h + 1
    W_out = (W + pad_left + pad_right - KW) // stride_w + 1
    assert(H_out == x.shape[2])
    assert(W_out == x.shape[3])

    dy_pad = np.pad(dy, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    w_trans = np.zeros((w.shape[1], w.shape[0], w.shape[2], w.shape[3]))
    for cout in range(Cout):
        for cin in range(Cin):
            for kh in range(KH):
                for kw in range(KW):
                    w_trans[cout, cin, kh, kw] = w[cin, cout, KH - 1 - kh, KW - 1 - kw]

    for nn in range(N):
        for nc in range(Cout):
            for nh in range(H_out):
                for nw in range(W_out):
                    dx[nn, nc, nh, nw] += np.sum(dy_pad[nn, :, nh * stride_h: nh * stride_h + KH, nw * stride_w: nw * stride_w + KW] * w_trans[nc, :, :, :], axis=(0, 1, 2))

    N, C, H, W = x.shape
    dx = dx.reshape(N, C // 16, 16, H, W).transpose(0, 1, 3, 4, 2).copy()

    return dx.astype(np.float16)


def gen_data(fmap_shape, filter_shape, pad_, stride_, dilation_, expect_file, attrs=None):
    block_size = 16
    in_n, in_c, in_h, in_w = fmap_shape
    cout, cin, w_h, w_w = filter_shape
    assert in_c == cin

    in_c = (in_c + block_size - 1) // block_size * block_size
    cout = (cout + block_size - 1) // block_size * block_size

    pad_top, pad_bottom, pad_left, pad_right = pad_
    stride_h, stride_w = stride_

    dilation_h, dilation_w = dilation_
    assert dilation_h == 1
    assert dilation_w == 1
    x_shape = (in_n, in_c, in_h, in_w)
    w_shape = (cout, in_c, w_h, w_w)

    p_top = w_h - pad_top - 1
    p_left = w_w - pad_left - 1
    p_bottom = in_h + pad_top - stride_h * ((in_h + pad_top + pad_bottom - w_h) // stride_h + 1)
    p_right = in_w + pad_left - stride_w * ((in_w + pad_left + pad_right - w_w) // stride_w + 1)

    print("Data gen ...")
    x = random_gaussian(x_shape, miu=1, sigma=0.1).astype(np.float16)
    w = random_gaussian(w_shape, miu=1, sigma=0.1).astype(np.float16)

    Ho = (x_shape[2] + pad_top + pad_bottom - w_shape[2]) // stride_h + 1
    Wo = (x_shape[3] + pad_left + pad_right - w_shape[3]) // stride_w + 1


    out_shape = (x_shape[0], w_shape[0], Ho, Wo)
    dout = random_gaussian(out_shape, miu=1, sigma=0.1).astype(np.float16)

    dx_shape = (in_n, in_c // block_size, in_h, in_w, block_size)
    flag_w = os.environ.get("WRITE_TO_DISK", "No")
    if (flag_w == "No") and (os.path.exists(expect_file) == True):
        # read expect from file
        dx = np.fromfile(expect_file, np.float16).reshape(dx_shape)
    else:
        # compute expect data:
        dx = conv_backprop_input_naive(x.astype(np.float32), w.astype(np.float32), dout.astype(np.float32),
                                       [p_top, p_bottom, p_left, p_right], [stride_h, stride_w])

    if flag_w == "Yes":
        # write expect to file
        with open(expect_file, "w+") as file:
            dx.tofile(file)
            file.close()

    # reshape
    C0 = block_size
    ON, OC, OH, OW = out_shape
    WN, WC, WH, WW = w_shape
    dout = dout.reshape(ON, OC // C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()
    w = w.reshape(WN, WC // C0, C0, WH, WW).transpose(1, 3, 4, 0, 2).copy()

    return dout, w, dx
