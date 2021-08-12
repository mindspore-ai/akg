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
import math
import numpy as np
from akg import tvm
from akg.utils import kernel_exec as utils
from akg.ops.nn import conv
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.test_run.conv_utils import conv_param_prepare, conv_shape_4d, conv_forward_naive, conv_tensor_4d_to_5d
from akg.utils import validation_check as vc_util
from akg.utils.kernel_exec import gen_kernel_name
from tests.common.test_utils import compute_blockdim


def conv_run(fmap_shape, filter_shape, pad, stride, dilation, use_bias=False, attrs=None, dump_data=False):
    conv_dtype = 'float16'

    vc_util.convolution_format_check(fmap_shape, filter_shape, pad, stride, dilation)

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    stride, pad, dilation = conv_param_prepare(conv_param)
    fm_shape, w_shape, out_shape = conv_shape_4d(fmap_shape, filter_shape, pad, stride, dilation)
    IN, IC, IH, IW = fm_shape
    WN, WC, WH, WW = w_shape
    C0 = 16

    if use_bias:
        input_shape = [(IN, IC // C0, IH, IW, C0), (WC // C0 * WH * WW, WN // 16, 16, C0), (1, WN // 16, 1, 1, 16)]
    else:
        input_shape = [(IN, IC // C0, IH, IW, C0), (WC // C0 * WH * WW, WN // 16, 16, C0)]

    input_file = os.environ.get("RANDOM_DATA_DISK_PATH", "")
    expect_file = input_file + "/" + gen_kernel_name([input_shape], [conv_dtype], op_attrs=[fmap_shape, filter_shape, pad, stride, dilation, use_bias, attrs],
                                                     kernel_name='conv', attrs=attrs) + ".bin"

    all_dynamic = 0      # kh kw pad stride
    partial_dynamic = 0  # fn fc1 fh fw wN wC
    if attrs.get("dynamic"):
        all_dynamic = 1
        print("=================all dynamic==================")
    if attrs.get("partial_dynamic"):
        partial_dynamic = 1
        print("=================partial dynamic==================")
    dynamic = partial_dynamic or all_dynamic

    if not dynamic:
        print("=================static shape==================")
    if dynamic:
        fmap_shape_real = fmap_shape
        filter_shape_real = filter_shape
        pad_real = pad
        stride_real = stride
        dilation_real = dilation

        if partial_dynamic or all_dynamic:
            N = tvm.var("N")
            C = tvm.var("CI")
            CI1 = tvm.var("CI1")
            H = tvm.var("H")
            W = tvm.var("W")

            COUT = tvm.var("CO")
            CO1 = tvm.var("CO1")
            _, _, KH, KW = filter_shape
            SH, SW = stride
            PT, PB, PL, PR = pad

        params = ()
        if all_dynamic:
            PARAM_KH = tvm.var("KH")
            PARAM_KW = tvm.var("KW") 
            PARAM_PT = tvm.var("PT")
            PARAM_PB = tvm.var("PB")
            PARAM_PL = tvm.var("PL")
            PARAM_PR = tvm.var("PR")
            PARAM_SH = tvm.var("SH")
            PARAM_SW = tvm.var("SW")

            PARAM_T1_0_H = tvm.var("T1_0_H")
            PARAM_T1_0_W = tvm.var("T1_0_W")
            PARAM_T1_0_C1 = tvm.var("T1_0_C1")
            PARAM_T0_0_MO = tvm.var("T0_0_MO") 
            PARAM_T0_0_NO = tvm.var("T0_0_NO")
            PARAM_T0_0_KO = tvm.var("T0_0_KO")

            params = (PARAM_KH, PARAM_KW, PARAM_PT, PARAM_PB, PARAM_PL, PARAM_PR, PARAM_SH, PARAM_SW,
                      PARAM_T1_0_H, PARAM_T1_0_W, PARAM_T1_0_C1, PARAM_T0_0_MO, PARAM_T0_0_NO, PARAM_T0_0_KO)

        DEBUG = 1
        if dynamic:
            KH_FAKE = 11
            KW_FAKE = 31
            fmap_shape = (N, C, H, W)
            filter_shape = (COUT, C, KH, KW)
            if not DEBUG:
                CO1 = (COUT + 15) // 16
                CI1 = (C + 15) // 16
            if use_bias:
                # input_shape = [(IN, IC // C0, IH, IW, C0), (WC // C0 * WH * WW, WN // 16, 16, C0), (1, WN // 16, 1, 1, 16)]
                if all_dynamic:
                    input_shape = [(N, CI1, H, W, 16), (CI1 * KH_FAKE * KW_FAKE, CO1, 16, 16), (1, CO1, 1, 1, 16)]
                else:
                    input_shape = [(N, CI1, H, W, 16), (CI1 * KH * KW, CO1, 16, 16), (1, CO1, 1, 1, 16)]
            else:
                # input_shape = [(IN, IC // C0, IH, IW, C0), (WC // C0 * WH * WW, WN // 16, 16, C0)]
                if all_dynamic:
                    input_shape = [(N, CI1, H, W, 16), (CI1 * KH_FAKE * KW_FAKE, CO1, 16, 16)]
                else:
                    input_shape = [(N, CI1, H, W, 16), (CI1 * KH * KW, CO1, 16, 16)]

        mod = utils.op_build_test(conv.conv, [input_shape], [conv_dtype],
                                  op_attrs=[fmap_shape, filter_shape, pad, stride, dilation, use_bias, attrs, params],
                                  kernel_name='conv', attrs=attrs)
        fmap_data, filter_data, bias_data, expect = gen_data(fmap_shape_real, filter_shape_real, pad_real, stride_real, dilation_real, use_bias, expect_file)
    else:
        mod = utils.op_build_test(conv.conv, [input_shape], [conv_dtype],
                                  op_attrs=[fmap_shape, filter_shape, pad, stride, dilation, use_bias, attrs],
                                  kernel_name='conv', attrs=attrs)
        fmap_data, filter_data, bias_data, expect = gen_data(fmap_shape, filter_shape, pad, stride, dilation, use_bias, expect_file)

    if dump_data:
        with open('input.bin', 'wb') as fo:
            fo.write(fmap_data.astype(np.float16, copy=False))
        with open('filter.bin', 'wb') as fo:
            fo.write(filter_data.astype(np.float16, copy=False))
        with open('bias.bin', 'wb') as fo:
            fo.write(bias_data.astype(np.float16, copy=False))
        with open('output.bin', 'wb') as fo:
            fo.write(expect.astype(np.float16, copy=False))

    out_data = np.full(expect.shape, np.nan, 'float16')

    if use_bias:
        input = [fmap_data, filter_data, bias_data]
    else:
        input = [fmap_data, filter_data]

    flag_w = os.environ.get("WRITE_TO_DISK", "No")
    if flag_w == "Yes":
        return input, out_data, expect, True

    if not dynamic:
        args = input
        args.append(out_data)
        args = tuple(args)
        out_data = utils.mod_launch(mod, args, expect=expect)
    else:
        args = []
        args.append(fmap_data)
        args.append(filter_data)
        args.append(out_data)
        if partial_dynamic or all_dynamic:
            args.append(IN)
            args.append(IC)
            args.append(IH)
            args.append(IW)
            args.append(WN)
        if all_dynamic:
            args.append(KH)
            args.append(KW)
            args.append(PT)
            args.append(PB)
            args.append(PL)
            args.append(PR)
            args.append(SH)
            args.append(SW)
            if attrs.get("conv_tile") and len(attrs["conv_tile"]) == 7:
                T1_0_H = attrs["conv_tile"][0]
                T1_0_C1 = attrs["conv_tile"][1]
                T0_0_MO = attrs["conv_tile"][2]
                T0_0_KO = attrs["conv_tile"][3]
                T0_0_NO = attrs["conv_tile"][4]
                T1_0_W = attrs["conv_tile"][5]
                if T1_0_H == IH:
                    T1_0_H += PT + PB
                T1_0_H_cut = (T1_0_H - KH) // SH + 1
                if T1_0_W == IW:
                    T1_0_W += PL + PR
                T1_0_W_cut = (T1_0_W - KW) // SW + 1
                args.append(T1_0_H_cut)
                args.append(T1_0_W_cut)
                args.append((T1_0_C1+15)//16)
                args.append((T0_0_MO+15)//16)
                args.append((T0_0_NO+15)//16)
                args.append((T0_0_KO+15)//16)
        if DEBUG:
            args.append(IC//16)
            args.append(WN//16)
        block_dim = min(32, IN)
        args.append(block_dim)
        out_data = utils.mod_launch(mod, args, outputs=(2,), expect=expect)

    rtol, atol = get_rtol_atol("conv", conv_dtype)
    return input, out_data, expect, compare_tensor(out_data, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(fm_shape, w_shape, pad, stride, dilation, bias, expect_file):

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    stride, pad, dilation = conv_param_prepare(conv_param)
    fm_shape, w_shape, out_shape = conv_shape_4d(fm_shape, w_shape, pad, stride, dilation)
    IN, IC, IH, IW = fm_shape
    WN, WC, WH, WW = w_shape

    x = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    w = random_gaussian((WN, WC, WH, WW), miu=0.5, sigma=0.01).astype(np.float16)

    if bias:
        b = random_gaussian((WN,), miu=1, sigma=0.1).astype(np.float16)
    else:
        b = (np.array(np.zeros(WN))).astype(np.float16, copy=False)

    flag_w = os.environ.get("WRITE_TO_DISK", "No")
    if (flag_w == "No") and (os.path.exists(expect_file)==True):
        #read expect from file
        out = np.fromfile(expect_file, np.float16).reshape(out_shape)
    else:
        #compute expect data:
        out = conv_forward_naive(x.astype(np.float32), w.astype(np.float32), b.astype(np.float32), conv_param)
        out = out.astype(np.float16)

    if flag_w == "Yes":
        # write expect to file
        with open(expect_file, "w+") as file:
            out.tofile(file)
            file.close()

    return conv_tensor_4d_to_5d(x, w, b, out)
