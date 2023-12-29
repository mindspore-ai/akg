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
from akg.ops.nn.ascend import conv_backprop_filter
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils.kernel_exec import gen_kernel_name
from tests.common.base import get_rtol_atol


def conv_backprop_filter_naive(x, w, y, pad_, stride_):
    N, C, H, W = x.shape
    _, _, OH, OW = y.shape
    CO, CI, KH, KW = w.shape

    pad_top, pad_bottom, pad_left, pad_right = pad_
    stride_h, stride_w = stride_
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                   constant_values=0)

    x_img2col = np.full((N, C, KH, KW, OH, OW), 0, np.float32)
    for nn in range(N):
        for nc in range(C):
            for nh in range(KH):
                for nw in range(KW):
                    for nho in range(OH):
                        for nwo in range(OW):
                            x_img2col[nn, nc, nh, nw, nho, nwo] = x_pad[
                                nn, nc, nho * stride_h + nh, nwo * stride_w + nw]

    dw = np.zeros_like(w)
    for nn in range(CO):
        for nc in range(CI):
            for nh in range(KH):
                for nw in range(KW):
                    dw[nn, nc, nh, nw] += np.sum(y[:, nn, :, :] * x_img2col[:, nc, nh, nw, :, :], axis=(0, 1, 2))

    N, C, H, W = dw.shape
    dw = dw.reshape(N, C // 16, 16, H, W).transpose(1, 3, 4, 0, 2).copy()
    dw = dw.reshape(C // 16 * H * W, N // 16, 16, 16)

    return dw


def gen_data(x_shape, w_shape, pad_, stride_, dilation_, expect_file, attrs=None):
    print("Data gen ...")
    block_size = 16

    x = random_gaussian(x_shape, miu=0.5, sigma=0.01).astype(np.float16)
    w = random_gaussian(w_shape, miu=1, sigma=0.1).astype(np.float16)

    pad_top, pad_bottom, pad_left, pad_right = pad_
    stride_h, stride_w = stride_
    dilation_h, dilation_w = dilation_
    Ho = (x_shape[2] + pad_top + pad_bottom - ((w_shape[2] - 1) * dilation_h + 1)) // stride_h + 1
    Wo = (x_shape[3] + pad_left + pad_right - ((w_shape[3] - 1) * dilation_w + 1)) // stride_w + 1

    out_shape = (x_shape[0], w_shape[0], Ho, Wo)
    y = random_gaussian(out_shape, miu=1, sigma=0.1).astype(np.float16)

    N, C, H, W = w_shape
    dw_shape = (C // block_size * H * W, N // block_size, block_size, block_size)
    flag_w = os.environ.get("WRITE_TO_DISK", "No")
    if (flag_w == "No") and (os.path.exists(expect_file) == True):
        # read expect from file
        dw = np.fromfile(expect_file, np.float32).reshape(dw_shape)
    else:
        # compute expect data:
        dw = conv_backprop_filter_naive(x.astype(np.float32), w.astype(np.float32), y.astype(np.float32), pad_, stride_)

    if flag_w == "Yes":
        # write expect to file
        with open(expect_file, "w+") as file:
            dw.tofile(file)
            file.close()

    # reshape
    C0 = block_size
    ON, OC, OH, OW = out_shape
    WN, WC, WH, WW = w_shape
    FN, FC, FH, FW = x_shape
    x = x.reshape(FN, FC // C0, C0, FH, FW).transpose(0, 1, 3, 4, 2).copy()
    y = y.reshape(ON, OC // C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()

    return y, x, dw

def conv_backprop_filter_run(fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=None):
    block_size = 16
    conv_dtype = 'float16'

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

    x_shape = (in_n, in_c, in_h, in_w)
    w_shape = (cout, in_c, w_h, w_w)
    y_shape = (out_n, out_c, out_h, out_w)
    inN, inC, inH, inW = x_shape
    input_shape_nc1hwc0 = (inN, inC // block_size, inH, inW, block_size)
    o_n, o_c, o_h, o_w = y_shape
    y_shape_nc1hwc0 = (o_n, o_c // block_size, o_h, o_w, block_size)

    input_shape = [y_shape_nc1hwc0, input_shape_nc1hwc0]

    input_file = os.environ.get("RANDOM_DATA_DISK_PATH", "")
    expect_file = input_file + "/" + gen_kernel_name([input_shape], [conv_dtype],
                              op_attrs=[fmap_shape, filter_shape, pad_, stride_, dilation_],
                              kernel_name='conv_backprop_filter', attrs=attrs) + ".bin"

    print("gen_data begin.")
    dy_data, dx_data, expect = gen_data(x_shape, w_shape, pad_, stride_, dilation_, expect_file, attrs=attrs)
    assert (dy_data.shape == y_shape_nc1hwc0)
    print("gen_data finished.")

    out_data = np.full(expect.shape, 0, 'float32')
    input = (dy_data, dx_data)

    flag_w = os.environ.get("WRITE_TO_DISK", "No")
    if flag_w == "Yes":
        return input, out_data, expect, True

    mod = utils.op_build_test(conv_backprop_filter, [input_shape], [conv_dtype],
                              op_attrs=[fmap_shape, filter_shape, pad_, stride_, dilation_],
                              kernel_name='conv_backprop_filter', attrs=attrs)
    args = (dy_data, dx_data, out_data)
    out_data = utils.mod_launch(mod, args, expect=expect)
    rtol, atol = get_rtol_atol("conv_backprop_filter", conv_dtype)
    return input, out_data, expect, compare_tensor(out_data, expect, rtol=rtol, atol=atol, equal_nan=True)
