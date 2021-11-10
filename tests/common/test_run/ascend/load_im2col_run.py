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
from akg.ops.nn.ascend import LoadIm2col
from akg.ops.nn.ascend.load_im2col import has_pad
from akg.utils.result_analysis import allclose_nparray
from tests.common.gen_random import random_gaussian

def load_im2col_run(fmap_shape, kernel_shape, stride, padding, dtype, attrs):
    block_size = 16
    fmap_n, fmap_c, fmap_h, fmap_w = fmap_shape
    if not (fmap_c % 16 == 0):
        raise RuntimeError("Channel needs to be divisible by 16 (needs padding) while channel is {}".format(fmap_c))
    fmap_shape_NC1HWCO = (fmap_n, fmap_c // block_size, fmap_h, fmap_w, block_size)
    mod = utils.op_build_test(LoadIm2col, [fmap_shape_NC1HWCO], [dtype], op_attrs = [kernel_shape, stride, padding], kernel_name='load_im2col', attrs=attrs)
    inputs, exp_output = gen_data(fmap_shape, kernel_shape, padding, stride, dtype)
    out_shape = gen_out_shape(fmap_shape, kernel_shape, padding, stride)
    output = np.full(out_shape, -2, dtype)
    acu_output = utils.mod_launch(mod,(inputs, output), expect=exp_output)

    debug = False
    if debug:
        allclose_nparray(exp_output, acu_output, rtol=5e-3, atol=5e-3)
    res = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)

    return inputs,acu_output,exp_output, res


def gen_out_shape(fmap_shape, kernel, pad, stride):
    n_,c_,h_,w_ = fmap_shape
    k_w, k_h = kernel
    stride_w, stride_h = stride
    pad_top, pad_bottom, pad_left, pad_right = pad
    ho = (h_ + pad_top + pad_bottom - k_h) // stride_h + 1
    wo = (w_ + pad_left + pad_right - k_w) // stride_w + 1
    m = ho * wo
    k = c_ * k_h * k_w
    out_shape = (n_, m//16, k//16, 16, 16)
    if m % 16 != 0 or has_pad(pad):
        m = n_ * m
        out_shape = (m//16, k//16, 16, 16)
    return out_shape

def gen_data(fmap_shape, kernel, pad, stride, dtype):
    N, C, H, W = fmap_shape
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel
    pad_t, pad_b, pad_l, pad_r = pad
    block_size = 16
    C0 = block_size
    C1 = C // block_size
    data_shape = (N, C1, H, W, C0)
    data = random_gaussian(data_shape, miu=1, sigma=0.1).astype(dtype)

    debug = False
    if debug:
        for index in range(H):
            for indexW in range(W):
                data[:, :, index, indexW, :] = (indexW + 1) + index * W
    Ho = (H + pad_b + pad_t - kernel_h) // stride_h + 1
    Wo = (W + pad_r + pad_l - kernel_w) // stride_w + 1

    data_pad_shape = (N, C1, H + pad_t + pad_b, W + pad_l + pad_r, C0)
    data_pad = np.zeros(data_pad_shape, dtype=dtype)
    data_pad[:, :, pad_t: pad_t + H, pad_l: pad_l + W, :] = data

    expect_shape = (N,
                    (Ho * Wo + block_size - 1) // block_size,
                    C1 * kernel_h * kernel_w,
                    block_size,
                    C0)
    m = Ho * Wo
    if m % 16 != 0 or has_pad(pad):
        expect_shape = (N * Ho * Wo // block_size,
                        C1 * kernel_h * kernel_w,
                        block_size,
                        C0)

    expect = np.zeros(expect_shape, dtype=dtype)

    if m % 16 == 0 and not(has_pad(pad)):
        for ho in range(Ho):
            for wo in range(Wo):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        for c1 in range(C1):
                            expect[:,
                            (ho * Wo + wo) // block_size,
                            kh * kernel_w * C1 + kw * C1 + c1,
                            (ho * Wo + wo) % block_size, :] \
                                = data_pad[:, c1, ho * stride_h + kh, wo * stride_w + kw, :]
    else:
        for n in range(N):
            for ho in range(Ho):
                for wo in range(Wo):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            for c1 in range(C1):
                                expect[(n * Ho * Wo + ho * Wo + wo) // block_size,
                                kh * kernel_w * C1 + kw * C1 + c1,
                                (n * Ho * Wo + ho * Wo+ wo) % block_size, :] \
                                    = data_pad[n, c1, ho * stride_h + kh, wo * stride_w + kw, :]

    return data, expect
