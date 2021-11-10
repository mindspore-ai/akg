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
from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
from tests.common.test_op.ascend import conv_ad_v2
from tests.common.test_run.ascend.conv_utils import conv_forward_naive
from tests.common.test_run.ascend.conv_utils import random_gaussian
from . import conv_filter_ad_run, conv_input_ad_run

def compare_5D(out_data, expect):
    return compare_tensor(out_data, expect, rtol=1e-3, atol=1e-3, equal_nan=True)


def conv_ad_v2_run(case_num, fmap_shape, filter_shape, pad_, stride_, dilation_,
                   use_bias=False, bypass_l1=False, dump_data=False, Tile=None, attrs=None):
    if Tile is None:
        Tile = [0, 0, 0, 0, 0]

    if (case_num == 1):
        mod_data, mod_weight = conv_ad_v2.conv_01(fmap_shape, filter_shape, pad_, stride_, dilation_,
                                                   tile_hh=Tile[0], tile_coco=Tile[1], tile_mm=Tile[2], tile_kk=Tile[3],
                                                   tile_nn=Tile[4], use_bias=False, block_size=16, conv_dtype='float16')
        in_n, in_c, in_h, in_w = fmap_shape
        k_n, k_c, k_h, k_w = filter_shape
        pad_h, pad_w, pad_l, pad_r = pad_
        s_h, s_w = stride_
        d_h, d_w = dilation_

        o_n = in_n
        o_c = k_n
        o_h = 1 + (in_h + 2 * pad_h - (k_h - 1) * d_h - 1) // s_h
        o_w = 1 + (in_w + 2 * pad_w - (k_w - 1) * d_w - 1) // s_w

        Head_strided_shape = (o_n, o_c, (o_h - 1) * s_h + 1, (o_w - 1) * s_w + 1)
        B_flip_shape = (k_c, k_n, k_h, k_w)
        fmap_data_01, filter_data_01, expect = gen_data(Head_strided_shape, B_flip_shape, k_h - 1, 1, 1)

        out_data_01 = np.full(expect.shape, 0, 'float16')
        input = (fmap_data_01, filter_data_01)
        args = (fmap_data_01, filter_data_01, out_data_01)
        out_data = utils.mod_launch(mod_data, args, expect=expect)

        assert_res = compare_5D(out_data, expect)

        return input, out_data, expect, assert_res

    elif (case_num == 2):
        mod_data, mod_weight,\
            mod_transpose_data, mod_transpose_convert_head,\
            mod_head_strided, mod_weight_flipped = conv_ad_v2.conv_02(fmap_shape, filter_shape, pad_, stride_, dilation_,
                                                                      tile_hh=Tile[0], tile_coco=Tile[1],
                                                                      tile_mm=Tile[2], tile_kk=Tile[3],
                                                                      tile_nn=Tile[4], bypass_l1=bypass_l1,
                                                                      use_bias=False, block_size=16,
                                                                      conv_dtype='float16')

        in_n, in_c, in_h, in_w = fmap_shape
        k_n, k_c, k_h, k_w = filter_shape
        pad_h, pad_w, pad_l, pad_r = pad_
        s_h, s_w = stride_
        d_h, d_w = dilation_
        block_size = 16

        o_n = in_n
        o_c = k_n
        o_h = 1 + (in_h + 2 * pad_h - (k_h - 1) * d_h - 1) // s_h
        o_w = 1 + (in_w + 2 * pad_w - (k_w - 1) * d_w - 1) // s_w


        # Test the kernel generated for backward_DATA
        Head_strided_shape = (o_n, o_c, (o_h - 1) * s_h + 1, (o_w - 1) * s_w + 1)
        B_flip_shape = (k_c, k_n, k_h, k_w)
        fmap_data_01, filter_data_01, expect = gen_data(Head_strided_shape, B_flip_shape, k_h - 1, 1, 1, strided=s_h)
        if (s_h <= 1):
            Head_origin_5D = fmap_data_01
        else:
            Head_origin_5D = np.full((o_n, o_c // block_size, o_h, o_w, block_size), 0, 'float16')
            for i0 in range(Head_origin_5D.shape[0]):
                for i1 in range(Head_origin_5D.shape[1]):
                    for i2 in range(Head_origin_5D.shape[2]):
                        for i3 in range(Head_origin_5D.shape[3]):
                            for i4 in range(Head_origin_5D.shape[4]):
                                Head_origin_5D[i0, i1, i2, i3, i4] = fmap_data_01[i0, i1, i2 * s_h, i3 * s_w, i4]
        B_origin_Fractal = np.flip(np.flip(filter_data_01, 1), 2)\
                             .reshape((k_n // block_size, k_h, k_w, k_c // block_size, block_size, block_size))
        B_origin_Fractal = np.transpose(B_origin_Fractal, (3, 1, 2, 0, 5, 4))\
                             .reshape((k_c // block_size * k_h * k_w, k_n // block_size, block_size, block_size))

        B_flipped_Fractal = np.reshape(filter_data_01, (k_n // block_size, k_h, k_w, k_c // block_size, block_size, block_size))
        B_flipped_Fractal = np.reshape(B_flipped_Fractal, (k_n // block_size * k_h * k_w, k_c // block_size, block_size, block_size))

        out_data_01 = np.full(expect.shape, 0, 'float16')
        input = (fmap_data_01, filter_data_01)
        args = (fmap_data_01, filter_data_01, out_data_01)
        out_data = utils.mod_launch(mod_data, args, expect=expect)

        assert_res = compare_5D(out_data, expect)

        H_strided = np.full((o_n, o_c // block_size, (o_h - 1) * s_h + 1, (o_w - 1) * s_w + 1, block_size), 0, 'float16')
        H_strided = utils.mod_launch(mod_head_strided, (Head_origin_5D, H_strided), expect=expect)

        B_flipped = np.full((k_n // block_size * k_h * k_w, k_c // block_size, block_size, block_size), 0, 'float16')
        B_flipped = utils.mod_launch(mod_weight_flipped, (B_origin_Fractal, B_flipped), expect=expect)

        assert_res &= compare_5D(H_strided, fmap_data_01)
        tmp1 = B_flipped_Fractal.reshape(-1).copy()
        tmp2 = B_flipped.reshape(-1).copy()
        ind = []
        for i in range(len(tmp1)):
            if (np.abs(tmp1[i] - tmp2[i]) > 0.05):
                ind.append(i)
        print("Len of bad indices: ", len(ind))
        assert_res &= (len(ind) == 0)
        print("Test result for backward_DATA = ", assert_res)

        # Test the kernel generated for backward_WEIGHT
        X_shape = (in_n, in_c, in_h, in_w)
        X_transposed_shape = (in_c, in_n, in_h, in_w)
        Head_shape = (o_n, o_c, o_h, o_w)
        Head_transposed_shape = (o_c, o_n, o_h, o_w)
        H_trans_fractal_shape = (o_c // block_size * o_h * o_w, o_n // block_size, block_size, block_size)

        fmap_data_02, filter_data_02, expect_02 = gen_data(X_transposed_shape, Head_transposed_shape, 0, 1, s_h)

        X_origin_5D = np.reshape(fmap_data_02, (in_c // block_size, block_size, in_n // block_size, in_h, in_w, block_size))\
                        .transpose((2, 5, 0, 3, 4, 1)).reshape((in_n, in_c // block_size, in_h, in_w, block_size))
        H_origin_5D = np.reshape(filter_data_02, (o_n // block_size, o_h, o_w, o_c // block_size, block_size, block_size))\
                        .transpose((0, 5, 3, 1, 2, 4)).reshape((o_n, o_c // block_size, o_h, o_w, block_size))

        out_data_02 = np.full(expect_02.shape, 0, 'float16')
        input_02 = (fmap_data_02, filter_data_02)
        args_02 = (fmap_data_02, filter_data_02, out_data_02)
        out_data_02 = utils.mod_launch(mod_weight, args_02, expect=expect_02)

        assert_res &= compare_5D(out_data_02, expect_02)

        X_trans = np.full(fmap_data_02.shape, 0, 'float16')
        X_trans = utils.mod_launch(mod_transpose_data, (X_origin_5D, X_trans))

        H_trans = np.full(H_trans_fractal_shape, 0, 'float16')
        H_trans = utils.mod_launch(mod_transpose_convert_head, (H_origin_5D, H_trans))

        Conv_trans = np.full(expect_02.shape, 0, 'float16')
        Conv_trans = utils.mod_launch(mod_weight, (X_trans, H_trans, Conv_trans))

        assert_res &= compare_5D(Conv_trans, expect_02)
        print("Test result for backward_DATA and WEIGHT = ", assert_res)

        return input_02, out_data_02, expect_02, assert_res

    elif (case_num == 3):
        return conv_input_ad_run.conv_input_ad_run(fmap_shape, filter_shape, pad_, stride_, dilation_)

    else:
        return conv_filter_ad_run.conv_filter_ad_run(fmap_shape, filter_shape, pad_, stride_, dilation_)


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
