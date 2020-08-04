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

import sys
import numpy as np

from akg.utils import kernel_exec as utils
from test_op import conv_mansch
from test_run.conv_utils import conv_forward_naive
from test_run.conv_utils import random_gaussian
from akg.utils.result_analysis import result_compare


def conv_run_mansch(FMap_shape, Filter_shape, Pad, Stride, Dilation=None, use_bias=False, bypass_L1=False,
                    dump_data=False, Tile=None, attrs=None):
    conv_dtype = 'float16'
    fp32_mad = True
    if attrs is not None and 'fp32mmad' in attrs:
        fp32_mad = attrs['fp32mmad']

    mod = conv_mansch.test_CCE_Conv(FMap_shape, Filter_shape, Pad, Stride,
                      Tile[0], Tile[1], Tile[2], Tile[3], Tile[4],
                      use_bias=use_bias, fp32_mad=fp32_mad, kernel_name="conv_mansch")

    source_code = mod.imported_modules[0].get_source()
    utils.create_code("conv_mansch", ".", source_code)
    A, B, bias_data, expect = gen_data(FMap_shape, Filter_shape, Pad, Stride, Dilation, use_bias)

    expect = expect.reshape((expect.shape[0], expect.shape[1], expect.shape[2]*expect.shape[3],expect.shape[4]))  # output on conv2d is in 4d format
    out_data = 60000.0*np.ones(expect.shape).astype(conv_dtype)
    if use_bias:
        out_data = utils.mod_launch(mod, [A.astype(conv_dtype), B.astype(conv_dtype), bias_data.astype(conv_dtype),
                                          out_data.astype(conv_dtype)], expect=expect)
    else:
        out_data = utils.mod_launch(mod, [A.astype(conv_dtype), B.astype(conv_dtype), out_data.astype(conv_dtype)],
                                    expect=expect)
    np.set_printoptions(threshold=sys.maxsize)
    assert_res = True
    try:
        assert_res = result_compare(out_data, expect, r_tol=5e-3)

        np.testing.assert_allclose(out_data, expect, rtol=5e-02, atol=1e-2, equal_nan=True, verbose=True)
        print("conv_test_Succeed")
    except BaseException as e:
        data_len = expect.size
        np.savetxt("actual.txt", out_data.reshape(data_len))
        np.savetxt("expect.txt", expect.reshape(data_len))
        print(str(e))

    return (A, B), out_data, expect, assert_res

def gen_data(fm_shape, w_shape, pad, stride, dilation, bias):

    if isinstance(stride,int):
        stride = [stride]*2
    elif isinstance(stride,(list,tuple)) and 1== len(stride):
        stride = list(stride)*2
    elif isinstance(stride,(list,tuple)) and 2== len(stride):
        pass
    else:
        raise RuntimeError('stride para illegal !!!')

    if isinstance(pad,int):
        pad = [pad]*4
    elif isinstance(pad,(list,tuple)) and 1==len(pad):
        pad = list(pad) *4
    elif isinstance(pad,(list,tuple)) and 4==len(pad):
        pass
    else:
        raise RuntimeError('pad para illegal !!!')

    if isinstance(dilation,int):
        dilation = [dilation]*2
    elif isinstance(dilation,(list,tuple)) and 1 == len(dilation):
        dilation = list(dilation)*2
    elif isinstance(dilation,(list,tuple)) and 2 == len(dilation):
        pass
    else:
        raise RuntimeError('dilation para illegal !!!')


    S_h,S_w = stride
    P_top,P_bottom,P_left,P_right = pad
    D_h,D_w = dilation


    IN, IC, IH, IW = fm_shape
    C0 = 16
    IC = ((IC+C0-1)//C0)*C0

    WN, WC, WH, WW = w_shape
    WN = ((WN+C0-1)//C0)*C0
    WC = ((WC+C0-1)//C0)*C0


    ON = IN
    OC = WN
    WHD = (WH - 1) * D_h + 1
    WWD = (WW - 1) * D_w + 1
    OH = (IH + P_top+P_bottom - WHD)//S_h + 1
    OW = (IW + P_left+P_right - WWD)//S_w + 1

    x = random_gaussian((IN, IC, IH, IW), miu=1, sigma=0.1).astype(np.float16)
    w = random_gaussian((WN, WC, WH, WW), miu=0.5, sigma=0.01).astype(np.float16)

    if bias:
        b = np.random.rand(WN).astype(np.float16, copy=False)
    else:
        b = (np.array(np.zeros(WN))).astype(np.float16, copy=False)

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    out = conv_forward_naive(x, w, b, conv_param)

    ''' transpose to 5D - NC1HWC0 '''
    feature = x.reshape(IN, IC//C0, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    ''' transpose to 5D - C1HWNC0 '''
    filter = w.reshape(WN, WC//C0, C0, WH, WW).transpose(1, 3, 4, 0, 2).copy()
    filter = filter.reshape(WC//C0*WH*WW, WN//16, 16,C0)

    bb = b.reshape(1,WN//16,1,1,16)
    ''' transpose to 5D - NC1HWC0 '''
    output = out.reshape(ON, OC//C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()

    return feature, filter, bb, output
