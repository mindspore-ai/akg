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
import shutil
import secrets
from collections import namedtuple
import numpy as np
import akg
import akg.tvm
import akg.lang.ascend
from akg import dim
from akg.utils import kernel_exec as utils
from tests.common.test_run.ascend.conv_utils import conv_forward_naive

secretsGenerator = secrets.SystemRandom()


def random_gaussian(size, miu=3, sigma=1):
    """ Generate random array with absolution value obeys gaussian distribution """
    if sigma <= 0:
        sys.stderr.write("Error: Expect positive sigmal for gaussian distribution. but get %f\n" % sigma)
        sys.exit(1)

    rgn = np.random.RandomState(2019)
    ret = rgn.normal(miu, sigma, size)
    for x in np.nditer(ret, op_flags=['readwrite']):
        if secretsGenerator.randint(0, 1):
            continue
        x[...] = x * -1
    return ret


def gen_data(fm_shape, w_shape, pad, stride, bias):
    IN, IC, IH, IW = fm_shape
    C0 = 16
    IC = ((IC + C0 - 1) // C0) * C0

    WN, WC, WH, WW = w_shape
    WN = ((WN + C0 - 1) // C0) * C0
    WC = ((WC + C0 - 1) // C0) * C0
    # WN = mt.ceil(WN/C0)*C0
    # WC = mt.ceil(WC/C0)*C0

    ON = IN
    OC = WN
    OH = (IH + 2 * pad - WH) // stride + 1
    OW = (IW + 2 * pad - WW) // stride + 1

    # np.random.seed(2)
    # x = ( np.random.rand(IN, IC, IH, IW) * 1.0 ).astype(np.float16, copy=False)
    # w = ( np.random.rand(WN, WC, WH, WW) - 0.5 ).astype(np.float16, copy=False)
    # b = ( np.array(np.zeros(WN)) ).astype(np.float16, copy=False)
    x = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    w = random_gaussian((WN, WC, WH, WW), miu=0.5, sigma=0.01).astype(np.float16)

    if bias:
        b = np.random.rand(WN).astype(np.float16, copy=False)
    else:
        b = (np.array(np.zeros(WN))).astype(np.float16, copy=False)

    # b = np.arange(WN).astype(np.float16, copy=False)
    # x = np.random.uniform(1, 1, size=(IN, IC, IH, IW)).astype(np.float16)
    # w = np.random.uniform(1, 1, size=(WN, WC, WH, WW)).astype(np.float16)
    # b = (np.array(np.ones(WN))).astype(np.float16, copy=False)
    # b = (np.array(np.full(WN, 9))).astype(np.float16, copy=False)

    conv_param = {'stride': stride, 'pad': pad}
    out = conv_forward_naive(x, w, b, conv_param)

    ''' transpose to 5D - NC1HWC0 '''
    feature = x.reshape(IN, IC // C0, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    ''' transpose to 5D - C1HWNC0 '''
    filter = w.reshape(WN, WC // C0, C0, WH, WW).transpose(1, 3, 4, 0, 2).copy()
    ''' transpose to 5D - NC1HWC0 '''
    output = out.reshape(ON, OC // C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()

    if fusion:
        zeros = np.full(output.shape, 0, output.dtype)
        output = np.maximum(zeros, output)

    return feature, filter, b, output


def run_conv(mod, fmap_shape, filter_shape, pad, stride, bias=False, dump_data=False):
    fmap_data, filter_data, bias_data, expect = gen_data(fmap_shape, filter_shape, pad, stride, bias)
    if dump_data:
        with open('input.bin', 'wb') as fo:
            fo.write(fmap_data.astype(np.float16, copy=False))
        with open('filter.bin', 'wb') as fo:
            fo.write(filter_data.astype(np.float16, copy=False))
        with open('bias.bin', 'wb') as fo:
            fo.write(bias_data.astype(np.float16, copy=False))
        with open('output.bin', 'wb') as fo:
            fo.write(expect.astype(np.float16, copy=False))

    # fmap_data = np.loadtxt('fuse_conv2d0_forword_data0_0.txt').reshape(fmap_shape).astype(np.float16)
    # filter_data = np.loadtxt('fuse_conv2d0_forword_kernel1_1.txt').reshape(filter_shape).astype(np.float16)
    # bias_data = np.loadtxt('fuse_conv2d0_forword_bias2_2.txt').reshape(filter_shape[0], ).astype(np.float16)

    out_data = np.full(expect.shape, 0, 'float16')

    rpc_ = utils.pandoraRpc()
    mod = rpc_.module_converter(mod)
    ctx = rpc_.gen_ctx()

    arg0 = akg.tvm.nd.array(fmap_data, ctx)
    arg1 = akg.tvm.nd.array(filter_data, ctx)
    out_arg = akg.tvm.nd.array(out_data, ctx)
    if bias:
        arg2 = akg.tvm.nd.array(bias_data, ctx)
        mod(arg0, arg1, arg2, out_arg)
    else:
        mod(arg0, arg1, out_arg)
    ctx.sync()

    # abs(output, expect) < 5*(10)^(-3) * abs(expect)
    data_len = expect.size
    try:
        actual = out_arg.asnumpy()
        # np.testing.assert_array_almost_equal(out_arg.asnumpy(), expect, 1)
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

                                # print "count: %6d expect: %10f actual: %10f %10.2f%%"%(count, b, a, abs((b-a)/b*100))

                            count += 1
        if continueErr > maxContinue:
            maxContinue = continueErr
            maxEnd = lastErr
        # print "error num: %d/%d (%.2f%%)" %(error, count, 100.0*error/count)
        # print "longest error range: [%d, %d]" %(maxEnd - maxContinue + 1, maxEnd)
        sys.stdout.flush()
        if maxContinue >= 16:
            os._exit(-1)
        np.testing.assert_allclose(actual, expect, rtol=5e-03, equal_nan=True, verbose=True)
        # print("\n\n******************** test ok *****************\n\n")
    except BaseException as e:
        np.savetxt("actual.txt", out_arg.asnumpy().reshape(data_len))
        np.savetxt("expect.txt", expect.reshape(data_len))
        # print(str(e))


fusion = False
run_cce = True

Conv_desc = namedtuple("Conv_desc",
                       ["in_n", "in_c", "in_h", "in_w", "cout", "w_h", "w_w",
                        "pad_left", "pad_right", "pad_top", "pad_bottom",
                        "stride_h", "stride_w", "bias",
                        "cutH", "cutCo", "cutM", "cutK", "cutN", "bypass_l1"])

resnet50_workload = [
    # 00 5m53.672s (mismatch 4.03180803571%) 71.1w cycle   fp32:6m34.740s (mismatch 0.00896843112245%) 71.7w cycle
    # Conv_desc(1  , 1024 , 14  , 14  , 2048 , 1  , 1  , 0 , 0 , 0 , 0 , 2 , 2, True, 14, 2048, 64, 96, 128, True),
    # 01 0m40.072s (mismatch 4.17530293367%) 4.6w cycle    fp32:0m51.811s (mismatch 0.00398596938776%) 4.7w cycle
    # Conv_desc(1  , 1024 , 14  , 14  , 256  , 1  , 1  , 0 , 0 , 0 , 0 , 1 , 1, True, 14, 256, 208, 64, 112, True),
    # 02 1m35.565s (mismatch 4.10554846939%) 19.3w cycle   fp32:1m47.179s (mismatch 0.00398596938776%) 19.5w cycle
    # Conv_desc(1  , 1024 , 14  , 14  , 512  , 1  , 1  , 0 , 0 , 0 , 0 , 2 , 2, True, 14, 512, 49, 32, 512, True),
    # 03 0m52.330s (mismatch 4.1334502551%) 5.1w cycle     fp32:1m3.842s (mismatch 0.00298947704081%) 5.1w cycle
    # Conv_desc(1  , 128  , 28  , 28  , 128  , 3  , 3  , 1 , 1 , 1 , 1 , 1 , 1, True, 28, 128, 400, 32, 128, False),
    # 04 0m55.584s (mismatch 1.43818757972%) 3.0w cycle    fp32:1m25.181s (mismatch 0.000996492346943%) 4.3w cycle
    # Conv_desc(1  , 128  , 28  , 28  , 512  , 1  , 1  , 0 , 0 , 0 , 0 , 1 , 1, True, 28, 512, 784, 16, 32, False),
    # 05 3m15.933s (mismatch 5.7955994898%) 36.2w cycle    fp32:3m2.766s (mismatch 0.0119579081633%) 36.3w cycle
    # Conv_desc(1  , 2048 , 7   , 7   , 512  , 1  , 1  , 0 , 0 , 0 , 0 , 1 , 1, True, 7, 512, 49, 32, 512, True),
    # 06 0m40.486s (mismatch 2.0358338648%) 3.2w cycle     fp32:0m57.671s (mismatch 0.000996492346943%) 3.6w cycle
    # Conv_desc(1  , 256  , 14  , 14  , 1024 , 1  , 1  , 0 , 0 , 0 , 0 , 1 , 1, True, 14, 944, 112, 32, 240, False),
    # 07 1m12.201s (mismatch 5.70591517857%) 9.2w cycle    fp32:1m17.662s (mismatch 0.00398596938776%) 9.2w cycle
    # Conv_desc(1  , 256  , 14  , 14  , 256  , 3  , 3  , 1 , 1 , 1 , 1 , 1 , 1, True, 14, 256, 196, 64, 256, True),
    # 08 0m35.593s (mismatch 1.98999521684%) 6.2w cycle    fp32:0m41.250s (mismatch 0.000996492346943%) 6.6w cycle
    # Conv_desc(1  , 256  , 56  , 56  , 128  , 1  , 1  , 0 , 0 , 0 , 0 , 2 , 2, True, 7, 128, 252, 64, 128, False),
    # 09 0m50.020s (mismatch 1.99398118622%) 5.3w cycle    fp32:1m3.908s (mismatch 0.00398596938776%) 5.5w cycle
    # Conv_desc(1  , 256  , 56  , 56  , 64   , 1  , 1  , 0 , 0 , 0 , 0 , 1 , 1, True, 16, 64, 280, 16, 64, False),
    # 10 4m47.902s (mismatch 3.4817442602%) 30.1w cycle    fp32:6m54.725s (mismatch 0.00510702327806%) 30.6w cycle
    # Conv_desc(1  , 3    , 224 , 224 , 64   , 7  , 7  , 3 , 3 , 3 , 3 , 2 , 2, True, 65, 64, 448, 32, 64, False),
    # 11 0m38.869s(mismatch 2.81110491071%) 4.0w cycle     fp32:1m12.797s (mismatch 0.000996492346943%) 4.3w cycle
    # Conv_desc(1  , 512  , 28  , 28  , 128  , 1  , 1  , 0 , 0 , 0 , 0 , 1 , 1, True, 14, 128, 448, 16, 64, False),
    # 12 0m28.570s (mismatch 2.78419961735%) 4.4w cycle    fp32:0m37.353s (mismatch 0.00398596938776%) 4.5w cycle
    # Conv_desc(1  , 512  , 28  , 28  , 256  , 1  , 1  , 0 , 0 , 0 , 0 , 2 , 2, True, 11, 256, 98, 64, 256, False),
    # 13 3m5.186s (mismatch 2.86092952806%) 34.2w cycle    fp32:3m54.894s (mismatch 0.00398596938776%) 34.8w cycle
    # Conv_desc(1  , 512  , 7   , 7   , 2048 , 1  , 1  , 0 , 0 , 0 , 0 , 1 , 1, True, 7, 2048, 49, 16, 512, True),
    # 14 4m0.641s (mismatch 7.72879464286%) 51.0w cycle    fp32:3m37.054s (mismatch 0.00797193877551%) 51.5w cycle
    # Conv_desc(1  , 512  , 7   , 7   , 512  , 3  , 3  , 1 , 1 , 1 , 1 , 1 , 1, True, 7, 512, 49, 32, 512, True),
    # 15 1m12.292s (mismatch 1.03909239477%) 5.4W cycle    fp32:2m2.360s (mismatch 0.00124561543367%) 7.4w cycle
    # Conv_desc(1  , 64   , 56  , 56  , 256  , 1  , 1  , 0 , 0 , 0 , 0 , 1 , 1, True, 56, 256, 784, 16, 32, False),
    # 16 0m21.902s (mismatch 1.05428890306%) 1.7w cycle    fp32:0m30.347s (mismatch 0.00149473852041%) 2.2w cycle
    Conv_desc(1, 64, 56, 56, 64, 1, 1, 0, 0, 0, 0, 1, 1, True, 56, 64, 784, 16, 32, False),
    # 17 1m6.587s (mismatch 3.01289461097%) 5.1w cycle     fp32:2m8.175s (mismatch 0.0049824617347%) 5.8w cycle
    # Conv_desc(1  , 64   , 56  , 56  , 64   , 3  , 3  , 1 , 1 , 1 , 1 , 1 , 1, True, 56, 64, 336, 16, 64, False), # 1008->336
    # 18 1m19.401s (mismatch 2.01266541773%) 11.3w cycle   fp32:1m34.275s (mismatch 0.000498246173464%) 12.0w cycle
    # Conv_desc(1  , 256  , 56  , 56  , 512   , 1 , 1  , 0 , 0 , 0 , 0 , 2 , 2, True, 7, 512, 196, 64, 256, False),
    # 19 3m13.809s (mismatch 2.86690848214%) 38.0w cycle   fp32:3m30.030s (mismatch 0.00348772321429%) 38.4w cycle
    # Conv_desc(1  , 512  , 28  , 28  , 1024  , 1 , 1  , 0 , 0 , 0 , 0 , 2 , 2, True, 13, 1024, 112, 32, 512, True),
]

################
block_size = 16
conv_dtype = 'float16'


def test_CCE_Conv(fmap_shape, filter_shape, pad_, stride_,
                  tile_hh=0, tile_coco=0, tile_mm=0, tile_kk=0, tile_nn=0, bypass_l1=False,
                  use_bias=False, kernel_name="quant_conv", cce_path='.'):
    # input shape (NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape
    input_shape_nc1hwc0 = (in_n, in_c // block_size, in_h, in_w, block_size)
    # out_shape_nc1hwc0 = (in_n, in_c // 32, in_h, in_w, 32)
    in_n, in_c1, in_h, in_w, in_c0 = input_shape_nc1hwc0

    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    kernel_shape_nc1hwc0 = (k_n, k_c // 32, k_h, k_w, 32)
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0
    kernel_shape_fractal = (k_c // 32 * k_h * k_w, k_n // 16, 16, 32)
    f_ko, f_no, f_ni, f_ki = kernel_shape_fractal

    # bias shape
    bias_shape_nc1hwc0 = (1, k_n // block_size, 1, 1, block_size)

    # padding ((padding_h, padding_w) -> (padding_top, padding_bottom, padding_left, padding_right))
    padding = (pad_[0], pad_[0], pad_[1], pad_[1])
    p_top, p_bottom, p_left, p_right = padding

    # stride (stride_h, stride_w)
    s_h, s_w = stride_

    # A placeholder (NC1HWCO)
    A = akg.tvm.placeholder(input_shape_nc1hwc0, dtype=conv_dtype, name='FMap')
    # B_placeholder (fractal)
    B = akg.tvm.placeholder(kernel_shape_fractal, dtype='int8', name='Filter')
    ScaleQ = akg.tvm.placeholder((16,), dtype='float16', name='ScaleQ')
    OffsetQ = akg.tvm.placeholder((16,), dtype='float16', name='OffsetQ')

    out_shape_nc1hwc0 = (in_n, in_c // 32, in_h, in_w, 32)
    q_n, q_c1, q_h, q_w, q_c0 = out_shape_nc1hwc0
    # print out_shape_nc1hwc0
    Quant = akg.tvm.compute(out_shape_nc1hwc0,
                            lambda qn, qc1, qh, qw, qc0: (
                                        A[qn, qc1 + qc0 // 16, qh, qw, qc0 % 16] * ScaleQ[0] + OffsetQ[0]).astype(
                                'int8'), name='QuantOUT', attrs={'no_inline': 1})

    if use_bias:
        bias_name = 'bias'
        bias_value = akg.tvm.placeholder(bias_shape_nc1hwc0, dtype=conv_dtype, name=bias_name)
    else:
        bias_name = 'None'

    # Create reduction variables
    kc1 = akg.tvm.reduce_axis((0, k_c1), name='kc1')
    kh = akg.tvm.reduce_axis((0, k_h), name='kh')
    kw = akg.tvm.reduce_axis((0, k_w), name='kw')
    kc0 = akg.tvm.reduce_axis((0, k_c0), name='kc0')

    out_h = (in_h + p_top + p_bottom - k_h) // (s_h) + 1
    tile_out_h = (tile_hh - k_h) // s_h + 1
    out_w = (in_w + p_left + p_right - k_w) // (s_w) + 1

    out_shape_nc1hwc0 = (in_n, k_n // block_size, out_h, out_w, block_size)
    out_n, out_c1, out_h, out_w, out_c0 = out_shape_nc1hwc0

    if (tile_coco > 0):
        c1_cut = tile_coco // block_size
    else:
        c1_cut = out_c1

    # set dim
    index = 0
    info = dim.Dim()
    if (q_c1 > 1):
        info.setdim(index=index, axis="KO", tilel1=q_c1, tilel0=q_c1)  # ko
    if (q_h > 1):
        info.setdim(index=index, axis="C1", tilel1=tile_out_h, tilel0=tile_out_h)  # c1
    if (q_w > 1):
        info.setdim(index=index, axis="C0", tilel1=q_w, tilel0=q_w)  # c0
    if (q_c0 > 1):
        info.setdim(index=index, axis="KI", tilel1=q_c0, tilel0=q_c0)  # ki

    index += 1
    if (out_c1 > 1):
        info.setdim(index=index, axis="C1", tilel1=c1_cut, tilel0=0)  # c1
    if (out_h > 1):
        info.setdim(index=index, axis="H", tilel1=tile_out_h, tilel0=0)  # h
    if (out_w > 1):
        info.setdim(index=index, axis="W", tilel1=out_w, tilel0=0)  # w
    if (out_c0 > 1):
        info.setdim(index=index, axis="C0", tilel1=out_c0, tilel0=0)  # c0
    if (in_c1 > 1):
        info.setdim(index=index, axis="KC1", tilel1=in_c1 / 2, tilel0=0)  # kc1
    if (k_h > 1):
        info.setdim(index=index, axis="KH", tilel1=k_h, tilel0=0)  # kh
    if (k_w > 1):
        info.setdim(index=index, axis="KW", tilel1=k_w, tilel0=0)  # kw
    info = str(info)

    # Compute the convolution
    output_name = "output0"
    output_bias_name = "output1"

    # print out_shape_nc1hwc0
    C = akg.tvm.compute(out_shape_nc1hwc0,
                        lambda n, c1, h, w, c0: akg.tvm.sum(
                            akg.tvm.if_then_else(
                                akg.tvm.any((h * s_h + kh) < p_top, (h * s_h + kh) > (in_h + p_top - 1),
                                            (w * s_w + kw) < p_left, (w * s_w + kw) > (in_w + p_left - 1)),
                                akg.tvm.const(0.0, 'int8'),
                                Quant[n, kc1, (h * s_h + kh - p_top), (w * s_w + kw - p_left), kc0])
                            * B[(kc1 * k_h + kh) * k_w + kw, c1, c0, kc0],
                            axis=[kc1, kh, kw, kc0]), name=output_name,
                        attrs={
                            "pragma_conv_kernel_n": k_n,
                            "pragma_conv_kernel_h": k_h,
                            "pragma_conv_kernel_w": k_w,
                            "pragma_conv_padding_top": p_top,
                            "pragma_conv_padding_bottom": p_bottom,
                            "pragma_conv_padding_left": p_left,
                            "pragma_conv_padding_right": p_right,
                            "pragma_conv_dilation_h": 1,
                            "pragma_conv_dilation_w": 1,
                            "pragma_conv_bypass_l1": 1 if bypass_l1 else 0,
                            "pragma_conv_stride_h": s_h,
                            "pragma_conv_stride_w": s_w,
                            "pragma_conv_fm_n": in_n,
                            "pragma_conv_fm_c": in_c,
                            "pragma_conv_fm_h": in_h,
                            "pragma_conv_fm_w": in_w,
                            "pragma_conv_h_cut": (h_window_cut - 1) * s_h + k_h,
                            "pragma_conv_w_cut": (in_w + p_left + p_right),
                            "pragma_conv_co_cut": c1_cut * k_c0,
                            "pragma_conv_m_cut": tile_mm,
                            "pragma_conv_k_cut": tile_kk,
                            "pragma_conv_n_cut": tile_nn,
                            "feature": Quant.op.name,
                            "filter": B.op.name,
                            "bias": bias_name,
                            "res": output_name,
                            "res_bias": output_bias_name})

    if use_bias:
        cube = akg.tvm.compute(out_shape_nc1hwc0,
                               lambda n, c1, h, w, c0: C[n, c1, h, w, c0] + bias_value[0, c1, 0, 0, c0],
                               name=output_bias_name)
    else:
        cube = C

    if fusion:
        # leakly relu
        negative_slope = 0.0
        slope_tmp = akg.tvm.const(negative_slope, dtype=conv_dtype)
        # negative_slope*x
        out = akg.lang.ascend.vmuls(cube, slope_tmp)
        # max(x,negative_slope*x)
        out = akg.lang.ascend.vmax(out, cube)
    else:
        out = cube

    # schedule
    s = akg.tvm.create_schedule(out.op)
    attrs = {}
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        if fusion:
            if use_bias:
                mod = akg.build(s, [A, B, ScaleQ, OffsetQ, bias_value, out], "cce", name=kernel_name,
                                attrs={"dim": info}, polyhedral=True)
            else:
                mod = akg.build(s, [A, B, ScaleQ, OffsetQ, out], "cce", name=kernel_name, attrs={"dim": info},
                                polyhedral=True)
        else:
            if use_bias:
                mod = akg.build(s, [A, B, ScaleQ, OffsetQ, bias_value, out], "cce", name=kernel_name,
                                attrs={"dim": info}, polyhedral=True)
            else:
                mod = akg.build(s, [A, B, ScaleQ, OffsetQ, out], "cce", name=kernel_name, attrs={"dim": info},
                                polyhedral=True)
    source_code = mod.imported_modules[0].get_source()
    # print(source_code)
    # utils.create_code(kernel_name, cce_path, source_code)
    if run_cce:
        run_conv(mod, fmap_shape, filter_shape, pad_[0], stride_[0], use_bias)


if __name__ == '__main__':
    count = 0
    for conv_layer in resnet50_workload:
        if len(sys.argv) == 2 and int(sys.argv[1]) != count:
            count += 1
            continue

        print("#########Run resnet50 Testcase %d:#########", count)
        count += 1

        # no l1 tiling is needed in bypass_l1 mode
        assert (conv_layer.bypass_l1 == False or conv_layer.cout == conv_layer.cutCo)

        in_n = conv_layer.in_n
        in_c = conv_layer.in_c
        in_c = (in_c + block_size - 1) // block_size * block_size
        in_h = conv_layer.in_h
        in_w = conv_layer.in_w

        cout = conv_layer.cout
        cout = (cout + block_size - 1) // block_size * block_size
        w_h = conv_layer.w_h
        w_w = conv_layer.w_w

        pad_left = conv_layer.pad_left
        pad_right = conv_layer.pad_right
        pad_top = conv_layer.pad_top
        pad_bottom = conv_layer.pad_bottom

        stride_h = conv_layer.stride_h
        stride_w = conv_layer.stride_w

        bias = conv_layer.bias

        cutH = conv_layer.cutH
        if (cutH == in_h):
            cutH += pad_top + pad_bottom
        cutCo = conv_layer.cutCo
        cutCo = (cutCo + block_size - 1) // block_size * block_size
        cutM = conv_layer.cutM
        cutM = (cutM + block_size - 1) // block_size * block_size
        cutK = conv_layer.cutK
        cutK = (cutK + block_size - 1) // block_size * block_size
        cutN = conv_layer.cutN
        cutN = (cutN + block_size - 1) // block_size * block_size

        c0 = block_size
        c1_cut = cutCo // c0
        h_window_cut = (cutH - w_h) // stride_h + 1

        out_w = (in_w + pad_left + pad_right - w_w) // (stride_w) + 1

        kernel_name = "quant_conv_layer_" + str(in_n) + "_" + str(in_c) + "_" + str(in_h) + "_" + str(in_w) \
                      + "_" + str(cout) + "_" + str(in_c) + "_" + str(w_h) + "_" + str(w_w) \
                      + "_" + str(pad_top) + "_" + str(stride_h)

        # kernel_name = "conv_layer_" + str(in_n) + "_" + str(in_c) + "_" + str(in_h) + "_" + str(in_w) \
        #                       + "_" + str(cout) + "_" + str(in_c) + "_" + str(w_h) + "_" + str(w_w) \
        #                       + "_" + str(pad_top) + "_" + str(pad_left) + "_" + str(stride_h) + "_" + str(stride_w) \
        #                       + "_" + ("1" if bias else "0")

        test_CCE_Conv((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (pad_top, pad_left), (stride_h, stride_w),
                      tile_hh=cutH, tile_coco=cutCo, tile_mm=cutM, tile_kk=cutK, tile_nn=cutN,
                      bypass_l1=conv_layer.bypass_l1,
                      use_bias=bias, kernel_name=kernel_name)

        if os.path.exists("conv.cce"):
            shutil.copyfile("conv.cce", kernel_name + ".cce")
