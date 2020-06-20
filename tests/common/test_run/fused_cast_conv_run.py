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
from akg.ops.nn import conv
from akg.ops.math import cast
from akg import dim
from test_run.conv_utils import conv_forward_naive
from test_run.conv_utils import random_gaussian
from akg.utils import custom_tiling as ct_util


cast_conv_set_dim_map = {
    # resnet50_wkl                                                                                tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww
    str(((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): ([14, 48, 64, 96, 128], {"bypass": 0}),  # 01
    str(((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([14, 32, 208, 64, 112], {"bypass": 0}),  # 02
    str(((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): ([14, 48, 49, 32, 512], {"bypass": 0}),  # 03
    str(((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): ([28, 32, 400, 32, 128], {"bypass": 0}),  # 04
    str(((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([28, 48, 784, 16, 32], {"bypass": 0}),  # 05
    str(((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([7, 16, 49, 32, 512], {"bypass": 0}),  # 06
    str(((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([14, 48, 112, 32, 240], {"bypass": 0}),  # 07
    str(((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): ([14, 16, 196, 64, 256], {"bypass": 0}),  # 08
    str(((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): ([7, 32, 252, 64, 128], {"bypass": 0}),  # 09
    str(((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([16, 64, 280, 16, 64], {"bypass": 0}),  # 10
    str(((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))): ([65, 48, 448, 32, 64], {"bypass": 0}),  # 11
    str(((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([14, 32, 448, 16, 64], {"bypass": 0}),  # 12
    str(((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): ([11, 32, 98, 64, 256], {"bypass": 0}),  # 13
    str(((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([7, 48, 49, 16, 512], {"bypass": 0}),  # 14
    str(((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): ([7, 16, 49, 32, 512], {"bypass": 0}),  # 15
    str(((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([56, 256, 784, 16, 32], {"bypass": 0}),  # 16
    str(((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): ([56, 64, 784, 16, 32], {"bypass": 0}),  # 17
    str(((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): ([56, 64, 336, 16, 64], {"bypass": 0}),  # 18
    str(((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): ([7, 128, 196, 64, 256], {"bypass": 0}),  # 19
    str(((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): ([13, 64, 112, 32, 512], {"bypass": 0}),  # 20
}


def cast_conv_set_dim_func(data, fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias=False, block_size=16, attrs=None):

    if isinstance(stride_, int):
        stride_ = [stride_] * 2
    elif isinstance(stride_, (list, tuple)) and 1 == len(stride_):
        stride_ = list(stride_) * 2
    elif isinstance(stride_, (list, tuple)) and 2 == len(stride_):
        pass
    else:
        raise RuntimeError('stride para illegal !!!')

    if isinstance(pad_, int):
        pad_ = [pad_] * 4
    elif isinstance(pad_, (list, tuple)) and 1 == len(pad_):
        pad_ = list(pad_) * 4
    elif isinstance(pad_, (list, tuple)) and 4 == len(pad_):
        pass
    else:
        raise RuntimeError('pad para illegal !!!')

    if isinstance(dilation_, int):
        dilation_ = [dilation_] * 2
    elif isinstance(dilation_, (list, tuple)) and 1 == len(dilation_):
        dilation_ = list(dilation_) * 2
    elif isinstance(dilation_, (list, tuple)) and 2 == len(dilation_):
        pass
    else:
        raise RuntimeError('dilation para illegal !!!')

    key = []

    key.append(tuple(fmap_shape))
    key.append(tuple(filter_shape))
    key.append(tuple(pad_))
    key.append(tuple(stride_))
    key.append(tuple(dilation_))

    hash_key = str(tuple(key))

    # input shape (NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape
    in_c = (in_c + block_size - 1) // block_size * block_size

    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    k_c = (k_c + block_size - 1) // block_size * block_size
    k_n = (k_n + block_size - 1) // block_size * block_size

    # padding((padding_h, padding_w) -> (padding_top, padding_bottom, padding_left, padding_right))
    padding = (pad_[0], pad_[0], pad_[1], pad_[1])
    p_top, p_bottom, p_left, p_right = padding

    # stride (stride_h, stride_w)
    s_h, s_w = stride_

    # dilation (dilation_h, dilation_w)
    d_h, d_w = dilation_

    k_w_d = (k_w - 1) * d_w + 1
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1
    bypass_list = [0, 1]
    bypass = 0
    if attrs is not None and 'conv_tile' in attrs and len(attrs['conv_tile']) >= 5:
        tile_hh = attrs['conv_tile'][0]
        tile_coco = attrs['conv_tile'][1]
        tile_mm = attrs['conv_tile'][2]
        tile_kk = attrs['conv_tile'][3]
        tile_nn = attrs['conv_tile'][4]
        if len(attrs['conv_tile']) > 5:
            tile_ww = attrs['conv_tile'][5]
        else:
            tile_ww = (out_w - 1) * s_w + k_w_d
        if 'bypass' in attrs:
            bypass = attrs['bypass']
    elif hash_key in cast_conv_set_dim_map:
        configs = cast_conv_set_dim_map[hash_key]
        if isinstance(configs, tuple):
            tiles = configs[0]
            if "bypass" in configs[1]:
                bypass = configs[1]["bypass"]
        else:
            tiles = configs
        if len(tiles) > 5:
            tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww = tiles
        else:
            tile_hh, tile_coco, tile_mm, tile_kk, tile_nn = tiles
            tile_ww = (out_w - 1) * s_w + k_w_d
    else:
        tile_hh = (k_h - 1) * d_h + 1 + p_top * s_h
        tile_ww = (out_w - 1) * s_w + k_w_d
        tile_coco = 16
        tile_mm = 16
        tile_kk = 16
        tile_nn = 16
    if not (bypass in bypass_list):
        raise RuntimeError("conv_cce ony supports %s while bypass is %d" % (",".join(str(bypass_list)), bypass))

    if (tile_hh == in_h):
        tile_hh += p_top + p_bottom
    tile_coco = (tile_coco + block_size - 1) // block_size * block_size
    tile_mm = (tile_mm + block_size - 1) // block_size * block_size
    tile_kk = (tile_kk + block_size - 1) // block_size * block_size
    tile_nn = (tile_nn + block_size - 1) // block_size * block_size

    c0 = block_size
    c1_cut = tile_coco // c0
    h_window_cut = (tile_hh - k_h) // s_h + 1

    out_w = (in_w + p_left + p_right - k_w) // (s_w) + 1

    input_shape_nc1hwc0 = (in_n, in_c // block_size, in_h, in_w, block_size)
    in_n, in_c1, in_h, in_w, in_c0 = input_shape_nc1hwc0

    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0

    k_h_d = (k_h - 1) * d_h + 1
    k_w_d = (k_w - 1) * d_w + 1
    out_h = (in_h + p_top + p_bottom - k_h_d) // (s_h) + 1
    tile_out_h = (tile_hh - k_h_d) // s_h + 1
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1
    tile_out_w = (tile_ww - k_w_d) // s_w + 1

    out_shape_nc1hwc0 = (in_n, k_n // block_size, out_h, out_w, block_size)
    out_n, out_c1, out_h, out_w, out_c0 = out_shape_nc1hwc0

    if (tile_coco > 0):
        c1_cut = tile_coco // block_size
    else:
        c1_cut = out_c1

    # set dim
    info = dim.Dim()
    if (out_n > 1):
        info.setdim(index=0, axis=0, tilel1=1, tilel0=0)  # n
    if (out_c1 > 1):
        info.setdim(index=0, axis=0, tilel1=c1_cut, tilel0=0)  # c1
    if (out_h > 1):
        info.setdim(index=0, axis="H", tilel1=tile_out_h, tilel0=0)  # h
    if (out_w > 1):
        info.setdim(index=0, axis="W", tilel1=tile_out_w, tilel0=0)  # w
    if (out_c0 > 1):
        info.setdim(index=0, axis=4, tilel1=out_c0, tilel0=0)  # c0

    if (in_c1 > 1):
        info.setdim(index=0, axis=5, tilel1=in_c1, tilel0=0)  # kc1
    if (k_h > 1):
        info.setdim(index=0, axis=5, tilel1=k_h, tilel0=0)  # kh
    if (k_w > 1):
        info.setdim(index=0, axis=5, tilel1=k_w, tilel0=0)  # kw

    return str(info)  # ct_util.set_dims_by_key(hash_key, conv_set_dim_map)


@ct_util.reg_set_dim_func(cast_conv_set_dim_func)
def cast_conv(data, fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias=False, block_size=16, attrs=None):
    a = data[0]
    data[1].dtype = 'float32'
    b = cast.cast(data[1], 'float16')
    if use_bias:
        conv_data = [a, b, data[2]]
    else:
        conv_data = [a, b]
    # mmad fp32 failed in post_fusing
    res, _ = conv.conv_core(conv_data, fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias, block_size, attrs)
    attr_map = {"pragma_reschedule": 1}
    return res, attr_map


def fused_cast_conv_run(fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias=False, dump_data=False, attrs=None):
    conv_dtype = 'float16'

    fmap_data, filter_data, bias_data, expect = gen_data(fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias)

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
        input = [fmap_data, filter_data, bias_data]
        input_shape = [fmap_data.shape, filter_data.shape, bias_data.shape]
    else:
        input = [fmap_data, filter_data]
        input_shape = [fmap_data.shape, filter_data.shape]

    args = input
    args.append(out_data)
    args = tuple(args)

    block_size = 16

    mod = utils.op_build_test(cast_conv, [input_shape], [conv_dtype], op_attrs=[fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias, block_size, attrs], kernel_name='cast_conv', attrs=attrs)

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

    return input, out_data, expect, assert_res


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

    x = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    w1 = random_gaussian((WN, WC, WH, WW), miu=0.5, sigma=0.01).astype(np.float32)
    w = w1.astype(np.float16)

    if bias:
        b = np.random.rand(WN).astype(np.float16, copy=False)
    else:
        b = (np.array(np.zeros(WN))).astype(np.float16, copy=False)

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    out = conv_forward_naive(x, w, b, conv_param)

    ''' transpose to 5D - NC1HWC0 '''
    feature = x.reshape(IN, IC // C0, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    ''' transpose to 5D - C1HWNC0 '''
    filter = w1.reshape(WN, WC // C0, C0, WH, WW).transpose(1, 3, 4, 0, 2).copy()
    filter = filter.reshape(WC // C0 * WH * WW, WN // 16, 16, C0)

    bb = b.reshape(1, WN // 16, 1, 1, 16)
    ''' transpose to 5D - NC1HWC0 '''
    output = out.reshape(ON, OC // C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()

    return feature, filter, bb, output
