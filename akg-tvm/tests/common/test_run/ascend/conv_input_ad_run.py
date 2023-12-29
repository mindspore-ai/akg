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

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.nn.ascend import ConvInputAd
from tests.common.test_run.ascend.conv_utils import conv_forward_naive
from tests.common.gen_random import random_gaussian
import os
from akg.utils.kernel_exec import gen_kernel_name
from tests.common.base import get_rtol_atol


def compare_5D(out_data, expect):
    data_len = expect.size
    actual = out_data
    N, C1, H, W, C0 = out_data.shape
    error = 0
    count = 0
    lastErr = -2
    continueErr = 0
    maxContinue = -1
    maxEnd = 0
    partial_debug = 0
    for n in range(N):
        for c1 in range(C1):
            for h in range(H):
                for w in range(W):
                    for c0 in range(C0):
                        a = actual[n, c1, h, w, c0]
                        b = expect[n, c1, h, w, c0]
                        if (abs(a - b) > abs(b) * 5e-02):
                            if (partial_debug and (a == 0.0)):
                                continue

                            error += 1
                            if lastErr + 1 == count:
                                continueErr += 1
                            else:
                                if continueErr > maxContinue:
                                    maxContinue = continueErr
                                    maxEnd = lastErr
                                continueErr = 1
                            lastErr = count
                        count += 1
    if continueErr > maxContinue:
        maxContinue = continueErr
        maxEnd = lastErr
    print("error num: %d/%d (%.2f%%)" % (error, count, 100.0 * error / count))
    print("longest error range: [%d, %d]" % (maxEnd - maxContinue + 1, maxEnd))

    if maxContinue >= 16:
        assert_res = False
    else:
        assert_res = True

    return assert_res


def conv_input_ad_run(fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=None):
    conv_dtype = 'float16'
    block_size = 16

    in_n, in_c, in_h, in_w = fmap_shape
    cout, cin, w_h, w_w = filter_shape
    assert(in_c == cin)

    in_c = (in_c + block_size - 1) // block_size * block_size
    cout = (cout + block_size - 1) // block_size * block_size

    pad_top, pad_bottom, pad_left, pad_right = pad_
    stride_h, stride_w = stride_

    out_n = in_n
    out_c = cout
    out_h = (in_h + pad_top + pad_bottom - w_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - w_w) // stride_w + 1

    w_shape = (cout, in_c, w_h, w_w)
    k_n, k_c, k_h, k_w = w_shape
    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0
    kernel_shape_fractal = (k_c // block_size * k_h * k_w, k_n // block_size, block_size, block_size)

    y_shape = (out_n, out_c, out_h, out_w)
    y_5D_shape = (out_n, out_c // block_size, out_h, out_w, block_size)

    dx_input_shapes = [y_5D_shape, kernel_shape_fractal]

    if attrs is None:
        attrs1 = dict()
    else:
        attrs1 = attrs.copy()
    attrs1["pragma_disable_whole_component"] = False
    input_file = os.environ.get("RANDOM_DATA_DISK_PATH", "")
    expect_file = input_file + "/" + gen_kernel_name([dx_input_shapes], [conv_dtype], op_attrs=[fmap_shape, filter_shape, pad_, stride_, dilation_, attrs],
                                                     kernel_name='conv_input_ad', attrs=attrs) + ".bin"

    print("gen_data begin.")
    fmap_data, filter_data, expect = gen_data_dx(fmap_shape, filter_shape, pad_, stride_, dilation_, expect_file, attrs=attrs1)
    print("gen_data finished.")

    out_data = np.full(expect.shape, 0, 'float16')
    np_input = (fmap_data, filter_data)

    flag_w = os.environ.get("WRITE_TO_DISK", "No")
    if flag_w == "Yes":
        return np_input, out_data, expect, True


    mod = utils.op_build_test(ConvInputAd, [dx_input_shapes], [conv_dtype],
                              op_attrs=[fmap_shape, filter_shape, pad_, stride_, dilation_, attrs],
                              kernel_name='conv_input_ad', attrs=attrs)
    args = (fmap_data, filter_data, out_data)
    out_data = utils.mod_launch(mod, args, expect=expect)
    rtol, atol = get_rtol_atol("conv_input_ad", conv_dtype)
    assert_res = compare_tensor(out_data, expect, rtol=rtol, atol=atol, equal_nan=True)

    return np_input, out_data, expect, assert_res

def gen_data(fm_shape, w_shape, pad, stride, dilation, strided=-1):
    IN, IC, IH, IW = fm_shape
    C0 = 16
    IC = ((IC + C0 - 1) // C0) * C0

    WN, WC, WH, WW = w_shape
    WN = ((WN + C0 - 1) // C0) * C0
    WC = ((WC + C0 - 1) // C0) * C0

    ON = IN
    OC = WN
    WHD = (WH - 1) * dilation + 1
    WWD = (WW - 1) * dilation + 1
    OH = (IH + 2 * pad - WHD) // stride + 1
    OW = (IW + 2 * pad - WWD) // stride + 1

    if (strided <= 1):
        x = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    else:
        x_tmp = random_gaussian((IN, IC, (IH // strided + 1), (IW // strided + 1)), miu=1, sigma=0.1).astype(np.float16)
        x = np.full((IN, IC, IH, IW), 0, dtype=np.float16)
        for i0 in range(x_tmp.shape[0]):
            for i1 in range(x_tmp.shape[1]):
                for i2 in range(x_tmp.shape[2]):
                    for i3 in range(x_tmp.shape[3]):
                        x[i0, i1, i2 * strided, i3 * strided] = x_tmp[i0, i1, i2, i3]

    w = random_gaussian((WN, WC, WH, WW), miu=0.5, sigma=0.01).astype(np.float16)

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    out = conv_forward_naive(x, w, None, conv_param)

    # transpose to 5D - NC1HWC0
    feature = x.reshape(IN, IC // C0, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    # transpose to 5D - C1HWNC0
    filter = w.reshape(WN, WC // C0, C0, WH, WW).transpose(1, 3, 4, 0, 2).copy()
    # transpose to 5D - NC1HWC0
    output = out.reshape(ON, OC // C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()

    return feature, filter, output


def calculate_conv_backprop_input(x, w, dy, pad_list, stride_list):
    N, C, H, W = dy.shape
    Cin, Cout, KH, KW = w.shape
    assert(C == Cin)

    pad_top, pad_bottom, pad_left, pad_right = pad_list
    stride_h, stride_w = stride_list

    if stride_h > 1 or stride_w > 1:
        dy_ = np.full((N, C, H * stride_h, W * stride_w), 0, np.float16)
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


def gen_data_dx(fmap_shape, filter_shape, pad_, stride_, dilation_, expect_file, attrs=None):
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
    b_shape = (w_shape[0], )

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
        dx = calculate_conv_backprop_input(x, w, dout, [p_top, p_bottom, p_left, p_right], [stride_h, stride_w])

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
