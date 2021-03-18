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
import numpy as np
import akg.topi
import akg.topi.testing
from akg.utils import kernel_exec as utils
from tests.common.test_op import group_conv_ad
import time
from tests.common.gen_random import random_gaussian

def compare(out_data, expect):
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

    if (100.0 * error / count) >= 5.0:
        assert_res = False
    else:
        assert_res = True

    return assert_res


def NCHW_to_5D(A, block_size):
    n, c, h, w = A.shape
    B = A.reshape((n, c // block_size, block_size, h, w)).transpose(0, 1, 3, 4, 2)
    return B


def NCHW_to_Fractal(A, block_size):
    n, c, h, w = A.shape
    B = A.reshape((n // block_size, block_size, c // block_size, block_size, h, w)).transpose(2, 4, 5, 0, 1, 3)\
         .reshape((c // block_size * h * w, n // block_size, block_size, block_size))
    return B


def group_conv_ad_run(N, H, W, CI, CO, group, KH, KW, PAD_H, PAD_W, SH, SW, cutH, cutCo, cutM, cutK, cutN, attrs):
    block_size = 16
    LOG_FILE_NAME = "log.txt"

    # set the environment to activate the backward Conv. Set to value "0" will deactivate the backward Conv
    os.environ["AD_CONV_ENABLE"] = "1"

    #OH, OW, A, B, C, mod =  depthwise_ad.depthwise_forward(N, H, W, CI, CO, group, KH, KW, PAD_H, PAD_W, SH, SW, cutH, cutCo, cutM, cutK, cutN, block_size)
    mod_data, mod_weights, mod_B_group_flip, mod_head_strided, mod_transposed_NC, mod_transposed_convert =\
        group_conv_ad.group_conv_ad(N, H, W, CI, CO, group, KH, KW, PAD_H, PAD_W, SH, SW, cutH, cutCo, cutM, cutK, cutN, block_size)

    OH = (H + 2 * PAD_H - KH) // SH + 1
    OW = (W + 2 * PAD_W - KW) // SW + 1

    CoG = CO // group
    CiG = CI // group

    np_H_NCHW = random_gaussian((N, CO, OH, OW), miu=1, sigma=0.1).astype(np.float16)
    np_H_5D = NCHW_to_5D(np_H_NCHW, block_size)
    assert_res = True

    # Testing the mini-kernel for striding the HEAD
    np_H_strided_NCHW = np.full((N, CO, (OH - 1) * SH + 1, (OW - 1) * SW + 1), 0, 'float16')

    for i0 in range(N):
        for i1 in range(CO):
            for i2 in range(OH):
                for i3 in range(OW):
                    np_H_strided_NCHW[i0, i1, i2 * SH, i3 * SW] = np_H_NCHW[i0, i1, i2, i3]
    np_H_strided_5D = NCHW_to_5D(np_H_strided_NCHW, block_size)
    out_H_strided = np.full(np_H_strided_5D.shape, 0, 'float16')

    print("Start launching StridingHead module...")
    start1 = time.time()
    out_H_strided = utils.mod_launch(mod_head_strided, [np_H_5D, out_H_strided])
    end1 = time.time()
    print("StridingHead with shape = ", np_H_5D.shape, " calculated in ", end1 - start1)
    print("MaxError of StridingHead = ", np.max(np.abs(out_H_strided - np_H_strided_5D)))
    assert_res &= (np.max(np.abs(out_H_strided - np_H_strided_5D)) < 1e-3)


    # Testing the mini-kernel for group-flipping the Weights
    np_B_NCHW = random_gaussian((CO, CiG, KH, KW), miu=0.5, sigma=0.01).astype(np.float16)
    np_B_flipped_NCHW = np.full((CI, CoG, KH, KW), 0, 'float16')
    for i0 in range(CI):
        for i1 in range(CoG):
            for i2 in range(KH):
                for i3 in range(KW):
                    np_B_flipped_NCHW[i0, i1, i2, i3] = np_B_NCHW[(i0 // CiG) * CoG + i1, i0 % CiG, KH - 1 - i2, KW - 1 - i3]

    np_B_fractal = NCHW_to_Fractal(np_B_NCHW, block_size)
    np_B_flipped_fractal = NCHW_to_Fractal(np_B_flipped_NCHW, block_size)
    out_B_flip = np.full(np_B_flipped_fractal.shape, 0, 'float16')

    print("Start launching GroupFlip module...")
    start1 = time.time()
    out_B_flip = utils.mod_launch(mod_B_group_flip, [np_B_fractal, out_B_flip])
    end1 = time.time()
    print("GroupFlip with shape = ", np_B_fractal.shape, " calculated in ", end1 - start1)
    print("MaxError of GroupFlip = ", np.max(np.abs(out_B_flip - np_B_flipped_fractal)))
    assert_res &= (np.max(np.abs(out_B_flip - np_B_flipped_fractal)) < 1e-3)


    # Testing the mini-kernel for transposing NC and regrouping the Data
    np_A_NCHW = random_gaussian((N, CI, H, W), miu=0.5, sigma=0.01).astype(np.float16)
    np_A_transposeNC_regroup_NCHW = np.transpose(np_A_NCHW, (1, 0, 2, 3)).reshape(group, CI // group, N, H, W)\
                                      .transpose(1, 0, 2, 3, 4).reshape(CI // group, group * N, H, W)

    np_A_5D = NCHW_to_5D(np_A_NCHW, block_size)
    np_A_transposeNC_regroup_5D = NCHW_to_5D(np_A_transposeNC_regroup_NCHW, block_size)
    out_A_transposeNC = np.full(np_A_transposeNC_regroup_5D.shape, 0, 'float16')

    print("Start launching TransposeNC module...")
    start1 = time.time()
    out_A_transposeNC = utils.mod_launch(mod_transposed_NC, [np_A_5D, out_A_transposeNC])
    end1 = time.time()
    print("TransposeNC with shape = ", np_A_5D.shape, " calculated in ", end1 - start1)
    print("MaxError of TransposeNC = ", np.max(np.abs(out_A_transposeNC - np_A_transposeNC_regroup_5D)))
    assert_res &= (np.max(np.abs(out_A_transposeNC - np_A_transposeNC_regroup_5D)) < 1e-3)


    # Testing the mini-kernel for transposing NC and converting to Fractal the HEAD
    np_H_transposeNC_NCHW = np.transpose(np_H_NCHW, (1, 0, 2, 3))
    np_H_transposeNC_fractal = NCHW_to_Fractal(np_H_transposeNC_NCHW, block_size)
    out_H_transposeNC_convert = np.full(np_H_transposeNC_fractal.shape, 0, 'float16')

    print("Start launching TransposeNC_convert module...")
    start1 = time.time()
    out_H_transposeNC_convert = utils.mod_launch(mod_transposed_convert, [np_H_5D, out_H_transposeNC_convert])
    end1 = time.time()
    print("TransposeNC_Convert with shape = ", np_H_5D.shape, " calculated in ", end1 - start1)
    print("MaxError of TransposeNC_convert = ", np.max(np.abs(out_H_transposeNC_convert - np_H_transposeNC_fractal)))
    assert_res &= (np.max(np.abs(out_H_transposeNC_convert - np_H_transposeNC_fractal)) < 1e-3)


    # Testing the kernel using GroupConv for gradient of Data
    start1 = time.time()
    expected1 = akg.topi.testing.conv2d_nchw_python(np_H_strided_NCHW, np_B_flipped_NCHW, (1, 1), (KH - 1 - PAD_H, KW - 1 - PAD_W), group)
    end1 = time.time()
    print("Expected dX with shape = ", expected1.shape, " calculated in ", end1 - start1)
    expected1_5D = NCHW_to_5D(expected1, block_size)
    out_dX = np.full(expected1_5D.shape, 0, 'float16')

    print("Start launching module...")
    start1 = time.time()
    out_dX = utils.mod_launch(mod_data, [out_H_strided, out_B_flip, out_dX], expect=expected1_5D)

    end1 = time.time()
    print("GroupConv2D for dX with shape = ", out_dX.shape, " calculated in ", end1 - start1)
    assert_res &= compare(out_dX, expected1_5D)


    # Testing the kernel using GroupConv for gradient of Weights
    start1 = time.time()
    expected2 = akg.topi.testing.conv2d_nchw_python(np_A_transposeNC_regroup_NCHW, np_H_transposeNC_NCHW, (SH, SW), (PAD_H, PAD_W), group)
    end1 = time.time()
    print("Expected dW with shape = ", expected2.shape, " calculated in ", end1 - start1)
    expected2_5D = expected2.reshape((CI // group, CO // block_size, block_size, KH, KW)).transpose([0, 1, 3, 4, 2])
    out_dW = np.full(expected2_5D.shape, 0, 'float16')

    print("Start launching module...")
    start1 = time.time()
    out_dW = utils.mod_launch(mod_weights, [out_A_transposeNC, out_H_transposeNC_convert, out_dW], expect=expected1_5D)
    end1 = time.time()
    print("GroupConv2D for dW with shape = ", out_dW.shape, " calculated in ", end1 - start1)

    assert_res &= compare(out_dW, expected2_5D)

    return (out_H_strided, out_B_flip), out_dX, expected1_5D, assert_res
