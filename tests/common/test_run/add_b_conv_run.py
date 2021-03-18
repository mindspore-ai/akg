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

import os
import sys

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op import add_b_conv
from tests.common.test_run.conv_utils import conv_forward_naive
from tests.common.test_run.conv_utils import random_gaussian


def add_b_conv_run(fmap_shape, filter_shape, pad_, stride_, dilation_,
                   use_bias=False, bypass_l1=False, dump_data=False, Tile=None, attrs=None):
    if Tile is None:
        Tile = [0, 0, 0, 0, 0]
    mod = add_b_conv.add_b_conv(fmap_shape, filter_shape, pad_, stride_, dilation_,
                                tile_hh=Tile[0], tile_coco=Tile[1], tile_mm=Tile[2], tile_kk=Tile[3], tile_nn=Tile[4], bypass_l1=bypass_l1,
                                use_bias=use_bias, block_size=16, conv_dtype='float16')
    fmap_data, filter_data, bias_data, expect = gen_data(fmap_shape, filter_shape, pad_[0], stride_[0], dilation_[0], use_bias)
    if dump_data:
        with open('input.bin', 'wb') as fo:
            fo.write(fmap_data.astype(np.float16, copy=False))
        with open('filter.bin', 'wb') as fo:
            fo.write(filter_data.astype(np.float16, copy=False))
        with open('bias.bin', 'wb') as fo:
            fo.write(bias_data.astype(np.float16, copy=False))
        with open('output.bin', 'wb') as fo:
            fo.write(expect.astype(np.float16, copy=False))

    out_data = np.full(expect.shape, 0, 'float16')
    if use_bias:
        input = (fmap_data, filter_data, bias_data)
        args = (fmap_data, filter_data, bias_data, out_data)
    else:
        input = (fmap_data, filter_data)
        args = (fmap_data, filter_data, out_data)
    out_data = utils.mod_launch(mod, args, expect=expect)

    # abs(output, expect) < 5*(10)^(-3) * abs(expect)
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

                                # if error < 35:
                                #     print ("count: %6d expect: %10f actual: %10f %10.2f%%"%(count, b, a, abs((b-a)/b*100)))

                            count += 1
        if continueErr > maxContinue:
            maxContinue = continueErr
            maxEnd = lastErr
        print("error num: %d/%d (%.2f%%)" % (error, count, 100.0 * error / count))
        print("longest error range: [%d, %d]" % (maxEnd - maxContinue + 1, maxEnd))
        sys.stdout.flush()
        if maxContinue >= 16:
            os._exit(-1)
        else:
            assert_res = True

        np.testing.assert_allclose(actual, expect, rtol=5e-03, equal_nan=True, verbose=True)
        print("\n\n******************** test ok *****************\n\n")
    except BaseException as e:
        np.savetxt("actual.txt", out_data.reshape(data_len))
        np.savetxt("expect.txt", expect.reshape(data_len))
        print(str(e))

    return input, out_data, expect, assert_res


def gen_data(fm_shape, w_shape, pad, stride, dilation, bias):
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

    x = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    w = random_gaussian((WN, WC, WH, WW), miu=0.5, sigma=0.01).astype(np.float16)
    w_add = w + 1.0

    if bias:
        b = np.random.rand(WN).astype(np.float16, copy=False)
    else:
        b = (np.array(np.zeros(WN))).astype(np.float16, copy=False)

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    out = conv_forward_naive(x, w_add, b, conv_param)

    ''' transpose to 5D - NC1HWC0 '''
    feature = x.reshape(IN, IC // C0, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    ''' transpose to 5D - C1HWNC0 '''
    filter = w.reshape(WN, WC // C0, C0, WH, WW).transpose(1, 3, 4, 0, 2).copy()
    ''' transpose to 5D - NC1HWC0 '''
    output = out.reshape(ON, OC // C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()

    return feature, filter, b, output
