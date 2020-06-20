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

import numpy as np
import akg.topi
import akg.topi.testing
from akg.utils import kernel_exec as utils
from test_op import depthwise
from gen_random import random_gaussian

def depthwise_run(N, H, W, CI, k_ch, KH, KW, PAD_H, PAD_W, SH, SW, attrs=None):
    conv_dtype = 'float16'
    block_size = 16
    LOG_FILE_NAME = "log.txt"
    CO = CI * k_ch
    group = CI // block_size

    bias_data = None

    A_np = random_gaussian((N, CI, H, W), miu=1, sigma=0.1).astype(np.float16)
    B_np = random_gaussian((CO, CI // group, KH, KW), miu=0.5, sigma=0.01).astype(np.float16)

    C_np = akg.topi.testing.conv2d_nchw_python(A_np, B_np, (SH, SW), (PAD_H, PAD_W), group)

    A_pack_np = A_np.reshape((N, CI // block_size, block_size, H, W)).transpose([0, 1, 3, 4, 2])

    B_shape = [CI // group // block_size * KH * KW, CO // block_size, block_size, block_size]
    B_pack_np = B_np.reshape((CO // block_size, block_size, CI // group // block_size, block_size, KH, KW)).transpose([2, 4, 5, 0, 1, 3]).reshape(B_shape)

    OH = (H + 2 * PAD_H - KH) // SH + 1
    OW = (W + 2 * PAD_W - KW) // SW + 1

    C_pack_np = C_np.reshape((N, CO // block_size, block_size, OH, OW)).transpose([0, 1, 3, 4, 2])

    out_np = np.full(C_pack_np.shape, 0, 'float16')

    input = [A_pack_np, B_pack_np]
    input_shape = [A_pack_np.shape, B_pack_np.shape]

    args = input
    args.append(out_np)
    args = tuple(args)

    if attrs is None:
        attrs = {}
    attrs["pragma_rmselfdep"] = False
    mod = utils.op_build_test(depthwise.depthwise, [input_shape], [conv_dtype], op_attrs=[N, H, W, CI, k_ch, KH, KW, PAD_H, PAD_W, SH, SW, block_size], kernel_name='depthwise', attrs=attrs)

    actual = utils.mod_launch(mod, args, expect=C_pack_np)

    expect = C_pack_np

    try:
        N, C1, H, W, C0 = out_np.shape
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
            print("error num: {0:d}/{1:d} ({2:.2f}%%".format(error, count,
                                                             100.0 * error / count))
            print("longest error range: [{0:d}, {1:d}]".format(maxEnd - maxContinue + 1,
                                                               maxEnd))

        if maxContinue >= 16:
            assert_res = False
        else:
            assert_res = True

        np.testing.assert_allclose(actual, expect, rtol=5e-01, equal_nan=True, verbose=True)
        msg = "All correct!"

    except Exception as e:

        msg = str(e)
        print(msg)

    with open(LOG_FILE_NAME, "a") as fout:
        fout.write(msg + "\n")

    return (A_np, B_np), actual, expect, assert_res
