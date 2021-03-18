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


def random_gaussian(size, miu=3, sigma=1):
    """ Generate random array with absolution value obeys gaussian distribution """
    if sigma <= 0:
        sys.stderr.write("Error: Expect positive sigmal for gaussian distribution. but get %f\n" % sigma)
        sys.exit(1)

    rgn = np.random.RandomState(2019)
    ret = rgn.normal(miu, sigma, size)
    for x in np.nditer(ret, op_flags=['readwrite']):
        if np.random.randint(0, 2):
            continue
        x[...] = x * -1
    return ret

def conv_param_prepare(conv_param):

    stride = 1
    pad = 0
    dilation = 1
    if 'stride' in conv_param:
        stride = conv_param['stride']
    if 'pad' in conv_param:
        pad = conv_param['pad']
    if 'dilation' in conv_param:
        dilation = conv_param['dilation']

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
    return stride, pad, dilation

def conv_shape_4d(fm_shape, w_shape, pad, stride, dilation):

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

    fm_shape = [IN, IC, IH, IW]
    w_shape = [WN, WC, WH, WW]
    out_shape = [ON, OC, OH, OW]

    return fm_shape, w_shape, out_shape

def conv_tensor_4d_to_5d(x, w, b, out):

    IN, IC, IH, IW = x.shape
    WN, WC, WH, WW = w.shape
    ON, OC, OH, OW = out.shape
    C0 =16

    ''' transpose to 5D - NC1HWC0 '''
    feature = x.reshape(IN, IC // C0, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    ''' transpose to 5D - C1HWNC0 '''
    filter = w.reshape(WN, WC // C0, C0, WH, WW).transpose(1, 3, 4, 0, 2).copy()
    filter = filter.reshape(WC // C0 * WH * WW, WN // 16, 16, C0)

    bb = b.reshape(1, WN // 16, 1, 1, 16)
    ''' transpose to 5D - NC1HWC0 '''
    output = out.reshape(ON, OC // C0, C0, OH, OW).transpose(0, 1, 3, 4, 2).copy()

    return feature, filter, bb, output

def conv_forward_naive(x, w, b, conv_param):

    stride, pad, dilation = conv_param_prepare(conv_param)

    P_top, P_bottom, P_left, P_right = pad[0:4]
    S_h, S_w = stride[0:2]
    D_h, D_w = dilation[0:2]

    fm_shape, w_shape, out_shape = conv_shape_4d(x.shape, w.shape, pad, stride, dilation)

    N, C, H, W = fm_shape
    WN, WC, WH, WW = w_shape
    _, F, Ho, Wo = out_shape
    WHD = (WH - 1) * D_h + 1
    WWD = (WW - 1) * D_w + 1

    x_pad = np.zeros((N, C, H + P_top + P_bottom, W + P_left + P_right))
    x_pad[:, :, P_top:P_top + H, P_left:P_left + W] = x
    out = np.zeros((N, F, Ho, Wo))

    for f in range(F):
        for i in range(Ho):
            for j in range(Wo):
                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                out[:, f, i, j] = np.sum(x_pad[:, :, i * S_h: i * S_h + WHD: D_h, j * S_w: j * S_w + WWD: D_w] * w[f, :, :, :], axis=(1, 2, 3))

        if b is not None:
            out[:, f, :, :] += b[f]

    return out
