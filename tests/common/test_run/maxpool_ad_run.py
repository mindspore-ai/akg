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

from tensorio import compare_tensor
import numpy as np
from akg.utils import kernel_exec as utils
from test_run import maxpool_grad_run
from akg.ops.nn.maxpool_ad import maxpool_ad
from akg.ops.nn.maxpool_ad import maxpool_ad_manual_schedule_all_max
from akg.ops.nn.maxpool_ad import maxpool_ad_no_custom_diff_manual_schedule_all_max
from akg.ops.nn.maxpool_ad import maxpool_ad_no_custom_diff_poly_all_max
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
import itertools
from base import get_rtol_atol
from gen_random import random_gaussian


def maxpool_ad_run(shape, kernel, stride, pad, dtype, optimized, polyhedral=False, first_max=True, attrs=None):
    expect, head, input, output, forward, mask = gen_data(dtype, kernel, pad, shape, stride, first_max)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)

    if polyhedral:
        if optimized:
            if first_max:
                raise Exception("ERROR: no DSL with poly support for first_max")
            else:
                raise Exception("ERROR: no DSL with poly support for all_max")
        else:
            if first_max:
                raise Exception("ERROR: no AD with poly support for first_max")
            else:
                mod = utils.op_build_test(maxpool_ad_no_custom_diff_poly_all_max, [head.shape, shape],
                                [dtype, dtype], kernel_name="maxpool_ad_no_custom_diff_poly_all_max",
                                op_attrs=[kernel, stride, pad], attrs=attrs, log_cce=False, dump_code=True, polyhedral=polyhedral)
                output = utils.mod_launch(mod, [head, input, output], expect=expect)
    else:
        if optimized:
            if first_max:
                mod = utils.op_build_test(maxpool_ad, [head.shape, shape, forward.shape, mask.shape],
                                        [dtype, dtype, dtype, dtype], kernel_name="maxpool_ad_first_max",
                                        op_attrs=[kernel, stride, pad], attrs=attrs, log_cce=False, dump_code=True, polyhedral=polyhedral)
                output = utils.mod_launch(mod, [head, input, forward, mask, output], expect=expect)
            else:
                mod = maxpool_ad_manual_schedule_all_max(shape, kernel, stride, pad, dtype, attrs=attrs, polyhedral=polyhedral)
                output = utils.mod_launch(mod, [head, input, forward, output], expect=expect)
        else:
            if first_max:
                raise Exception("ERROR: no AD with mansch support for first_max")
            else:
                mod = utils.op_build_test(maxpool_ad_no_custom_diff_manual_schedule_all_max, [head.shape, shape],
                                [dtype, dtype], kernel_name="maxpool_ad_no_custom_diff_manual_schedule_all_max",
                                op_attrs=[kernel, stride, pad], attrs=attrs, log_cce=False, dump_code=True, polyhedral=polyhedral)
                output = utils.mod_launch(mod, [head, input, output], expect=expect)
    
    if 'tuning' in attrs.keys():
        if t:
            return mod, expect, (head, input, output)
        else:
            return mod

    rtol, atol = get_rtol_atol("maxpool_grad", dtype)
    return [head, input], output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)

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
    mask = np.zeros((N, C1, kernel_h, kernel_w, yH, yW, C0))

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
                                        mask[n, c1, kh, kw, yh, yw, c0] = 1.0
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

    return dx[:, :, pad_h_head:(pad_h_head + H), pad_w_head:(pad_w_head + W), :], mask


def gen_data(dtype, kernel, pad, shape, stride, first_max):
    if first_max:
        behaviour = 0
    else:
        behaviour = 1
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    y = maxpool_grad_run.maxpool_benchmark(input, kernel, stride, pad).astype(dtype)
    head = random_gaussian(y.shape, miu=1, sigma=0.1).astype(dtype)
    expect, mask = benchmark(input, y, head, kernel, stride, pad, behaviour)
    expect = expect.astype(dtype)
    mask = mask.astype(dtype)
    out_shape = expect.shape
    output = np.full(out_shape, 0.0, dtype)
    return expect, head, input, output, y, mask
