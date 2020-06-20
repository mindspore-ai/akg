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
import itertools
from akg.utils import kernel_exec as utils
from akg.ops.nn import maxpool_grad
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian


def maxpool_benchmark(input, kernel, stride, pad):
    sh, sw = stride
    [ph_h, ph_t, pw_h, pw_t], [out_size_h, out_size_w] = \
        cal_pad_shapes_by_strategy(input.shape, kernel, stride, pad)
    N, C1, H, W, C0 = input.shape
    KH, KW = kernel

    out_shape = (N, C1, out_size_h, out_size_w, C0)

    out = np.zeros(out_shape)

    inputpad = np.full((N, C1, H + ph_h + ph_t, W + pw_h + pw_t, C0),
                       np.finfo(input.dtype).min, dtype=input.dtype)
    inputpad[:, :, ph_h: ph_h + H, pw_h: pw_h + W, :] = input

    for i in range(out_size_h):
        for j in range(out_size_w):
            out[:, :, i, j, :] = np.max(inputpad[:, :,
                                                 i * sh:i * sh + KH, j * sw:j * sw + KW, :], axis=(2, 3))
    return out


# behaviour
#   - 0: return dy only for the first maximum within one kernel
#   - 1: return dy for all maximums within one kernel
def benchmark(x, y, dy, kernel, stride, pad, behaviour=0):

    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    [pad_h_head, pad_h_tail, pad_w_head, pad_w_tail], _ = cal_pad_shapes_by_strategy(x.shape, kernel, stride, pad)
    N, C1, H, W, C0 = x.shape
    pad_shape = (N, C1, H + pad_h_tail + pad_h_head, W + pad_w_tail + pad_w_head, C0)

    padx = np.full(pad_shape, 0, dtype=x.dtype)
    padx[:, :, pad_h_head:(pad_h_head + H), pad_w_head:(pad_w_head + W), :] = x

    dx = np.zeros(padx.shape, dtype=x.dtype)
    _, _, yH, yW, _ = y.shape

    if behaviour == 0:
        for n in range(N):
            for c1 in range(C1):
                for yh in range(yH):
                    for yw in range(yW):
                        for c0 in range(C0):
                            out_maxpool1 = y[n, c1, yh, yw, c0]
                            head_maxpool1 = dy[n, c1, yh, yw, c0]
                            for kh,kw in itertools.product(range(kernel_h), range(kernel_w)):
                                    if padx[n, c1, yh*stride_h + kh, yw*stride_w + kw, c0] == out_maxpool1:
                                        dx[n, c1, yh*stride_h + kh, yw*stride_w + kw, c0] += head_maxpool1
                                        break
    elif behaviour == 1:
        for n in range(N):
            for c1 in range(C1):
                for yh in range(yH):
                    for yw in range(yW):
                        for c0 in range(C0):
                            out_maxpool1 = y[n, c1, yh, yw, c0]
                            head_maxpool1 = dy[n, c1, yh, yw, c0]
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    if padx[n, c1, yh*stride_h + kh, yw*stride_w + kw, c0] == out_maxpool1:
                                        dx[n, c1, yh*stride_h + kh, yw*stride_w + kw, c0] += head_maxpool1

    return dx[:, :, pad_h_head:(pad_h_head + H), pad_w_head:(pad_w_head + W), :]


def maxpool_grad_run(shape, kernel, stride, pad, dtype, attrs):
    # Create op
    _, [yh, yw] = cal_pad_shapes_by_strategy(shape, kernel, stride, pad)
    y_shape = (shape[0], shape[1], yh, yw, shape[4])

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(maxpool_grad.maxpool_grad,
                                  [shape, y_shape, y_shape], [dtype, dtype, dtype],
                                  op_attrs=[kernel, stride, pad],
                                  kernel_name=kernel_name, attrs=attrs, dump_cce=True, tuning=t)
        if t:
            dy, expect, output, x, y = \
                gen_data(dtype, kernel, pad, shape, stride, y_shape)
            return mod, expect, (x, y, dy, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(maxpool_grad.maxpool_grad,
                                  [shape, y_shape, y_shape], [dtype, dtype, dtype],
                                  op_attrs=[kernel, stride, pad],
                                  kernel_name='maxpool_grad', attrs=attrs, dump_cce=True)
        dy, expect, output, x, y = \
            gen_data(dtype, kernel, pad, shape, stride, y_shape)
        output = utils.mod_launch(mod, (x, y, dy, output), expect=expect)
        rtol, atol = get_rtol_atol("maxpool_grad", dtype)
        return (x, y, dy), output, expect, \
            compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, kernel, pad, shape, stride, y_shape):
    # Generate data for testing the op
    x = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    y = maxpool_benchmark(x, kernel, stride, pad)
    y = y.astype(dtype)
    assert y.shape == y_shape, "y shape wrong"
    dy = random_gaussian(y_shape, miu=1, sigma=0.1).astype(dtype)
    expect = benchmark(x, y, dy, kernel, stride, pad)
    output_shape = shape
    # output tensor must be initialized to zero
    output = np.full(output_shape, 0, dtype)
    return dy, expect, output, x, y
