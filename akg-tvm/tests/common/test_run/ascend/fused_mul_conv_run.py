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
import sys

import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.nn.ascend import Conv
from akg.ops.math import mul
from tests.common.test_run.ascend.conv_utils import conv_forward_naive
from tests.common.test_run.ascend.conv_utils import random_gaussian


def mul_conv(data, fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias=False,
             block_size=16, attrs=None, target="cce"):
    a1 = data[0]
    a2 = data[1]
    b = data[2]
    a = mul(data[0], data[1], target=target)
    if use_bias:
        conv_data = [a, b, data[3]]
    else:
        conv_data = [a, b]
    res = Conv(conv_data, fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias, block_size, attrs)
    return res


def fused_mul_conv_run(fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias=False, dump_data=False, attrs=None):
    conv_dtype = 'float16'

    fmap_data1, fmap_data2, filter_data, bias_data, expect = gen_data(fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias)

    if dump_data:
        with open('input1.bin', 'wb') as fo:
            fo.write(fmap_data1.astype(np.float16, copy=False))
        with open('input2.bin', 'wb') as fo:
            fo.write(fmap_data2.astype(np.float16, copy=False))
        with open('filter.bin', 'wb') as fo:
            fo.write(filter_data.astype(np.float16, copy=False))
        with open('bias.bin', 'wb') as fo:
            fo.write(bias_data.astype(np.float16, copy=False))
        with open('output.bin', 'wb') as fo:
            fo.write(expect.astype(np.float16, copy=False))

    out_data = np.full(expect.shape, 0, 'float16')

    if use_bias:
        input_ = [fmap_data1, fmap_data2, filter_data, bias_data]
        input_shape = [fmap_data1.shape, fmap_data2.shape, filter_data.shape, bias_data.shape]
    else:
        input_ = [fmap_data1, fmap_data2, filter_data]
        input_shape = [fmap_data1.shape, fmap_data2.shape, filter_data.shape]

    args = input_
    args.append(out_data)
    args = tuple(args)

    block_size = 16

    mod = utils.op_build_test(mul_conv, [input_shape], [conv_dtype], op_attrs=[fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias, block_size, attrs], kernel_name='mul_conv', attrs=attrs)

    out_data = utils.mod_launch(mod, args, expect=expect)

    data_len = expect.size
    try:
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
                            if (abs(a - b) > abs(b) * 5e-03):
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
        sys.stdout.flush()
        if maxContinue >= 16:
            assert_res = False
        else:
            assert_res = True

        np.testing.assert_allclose(actual, expect, rtol=5e-02, atol=1e-2, equal_nan=True, verbose=True)
        print("\n\n******************** test ok *****************\n\n")
    except BaseException as e:
        np.savetxt("actual.txt", out_data.reshape(data_len))
        np.savetxt("expect.txt", expect.reshape(data_len))
        print(str(e))

    return input_, out_data, expect, assert_res


def gen_data(fm_shape, w_shape, pad, stride, dilation, bias):

    if isinstance(stride, int):
        stride = [stride] * 2
    elif isinstance(stride, (list, tuple)) and 1 == len(stride):
        stride = list(stride) * 2
    elif isinstance(stride, (list, tuple)) and 2 == len(stride):
        pass
    else:
        raise RuntimeError('stride para illegal !!!')

    if isinstance(pad, int):
        pad = [pad] * 4
    elif isinstance(pad, (list, tuple)) and 1 == len(pad):
        pad = list(pad) * 4
    elif isinstance(pad, (list, tuple)) and 4 == len(pad):
        pass
    else:
        raise RuntimeError('pad para illegal !!!')

    if isinstance(dilation, int):
        dilation = [dilation] * 2
    elif isinstance(dilation, (list, tuple)) and 1 == len(dilation):
        dilation = list(dilation) * 2
    elif isinstance(dilation, (list, tuple)) and 2 == len(dilation):
        pass
    else:
        raise RuntimeError('dilation para illegal !!!')

    S_h, S_w = stride
    P_top, P_bottom, P_left, P_right = pad
    D_h, D_w = dilation

    IN, IC, IH, IW = fm_shape
    C0 = 16
    IC = ((IC + C0 - 1) // C0) * C0

    WN, WC, WH, WW = w_shape
    WN = ((WN + C0 - 1) // C0) * C0
    WC = ((WC + C0 - 1) // C0) * C0

    ON = IN
    OC = WN
    WHD = (WH - 1) * D_h + 1
    WWD = (WW - 1) * D_w + 1
    OH = (IH + P_top + P_bottom - WHD) // S_h + 1
    OW = (IW + P_left + P_right - WWD) // S_w + 1

    x1 = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    x2 = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    x = fmap_data = np.multiply(x1, x2)
    w = random_gaussian((WN, WC, WH, WW), miu=0.5, sigma=0.01).astype(np.float16)

    if bias:
        b = np.random.rand(WN).astype(np.float16, copy=False)
    else:
        b = (np.array(np.zeros(WN))).astype(np.float16, copy=False)

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    out = conv_forward_naive(x, w, b, conv_param)

    # transpose to 5D - NC1HWC0
    feature1 = x1.reshape(IN, IC // C0, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    feature2 = x2.reshape(IN, IC // C0, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    # transpose to 5D - C1HWNC0
    filter_ = w.reshape(WN, WC // C0, C0, WH, WW).transpose(1, 3, 4, 0, 2).copy()
    filter_ = filter_.reshape(WC // C0 * WH * WW, WN // 16, 16, C0)

    bb = b.reshape(1, WN // 16, 1, 1, 16)
    # transpose to 5D - NC1HWC0
    output = out.reshape(ON, OC // C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()

    return feature1, feature2, filter_, bb, output
