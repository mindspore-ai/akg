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
import akg
import akg.tvm
import akg.topi
from akg.utils import kernel_exec as utils
from test_op.winograd_ad import winograd_ad
from tensorio import compare_tensor


def winograd_ad_run(filter_shape, tile, dtype, attrs):
    def RANGEFILL(shape):
        size = np.prod([d for d in shape])
        offset = size // 2
        return np.arange(-offset, size - offset, dtype=dtype).reshape(shape)

    A = akg.tvm.placeholder(filter_shape, dtype=dtype, name='image')
    B = akg.topi.nn.conv2d_winograd_weight_transform(A, 2)
    head_np = RANGEFILL([d.value for d in B.shape]).astype(dtype)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(winograd_ad, [head_np.shape, filter_shape], [dtype, dtype], kernel_name=kernel_name,
                                  attrs=attrs, log_cce=True, dump_code=True, tuning=t)
        if t:
            expect, input_np, output = gen_data(filter_shape, RANGEFILL, dtype)
            return mod, expect, (head_np, input_np, output)
        else:
            return mod
    else:
        # scenario 1:
        expect, input_np, output = gen_data(filter_shape, RANGEFILL, dtype)
        mod = utils.op_build_test(winograd_ad, [head_np.shape, filter_shape], [dtype, dtype], kernel_name="winograd_ad",
                                  attrs=attrs, log_cce=True, dump_code=True)
        output = utils.mod_launch(mod, [head_np, input_np, output], expect=expect)
        if not compare_tensor(output, expect, atol=0.1):
            return [head_np, input_np], output, expect, compare_tensor(output, expect, rtol=5e-03, atol=5e-03,
                                                                       equal_nan=True)

        # scenario 2:
        head_np = np.ones(np.prod([d for d in B.shape])).reshape(B.shape).astype(dtype)
        expect = np.array(
            [[[4., 0., 4.], [0., 0., 0.], [4., 0., 4.]], [[4., 0., 4.], [0., 0., 0.], [4., 0., 4.]]]).astype(dtype)
        return [head_np, input_np], output, expect, compare_tensor(output, expect, rtol=5e-03, atol=5e-03,
                                                                   equal_nan=True)


def gen_data(filter_shape, RANGEFILL, dtype):
    input_np = RANGEFILL(filter_shape).astype(dtype)
    expect = np.array([[[-34., -2., -22.], [-8., 0., -8.], [14., -2., 26.]],
                       [[-30., -2., -18.], [-8., 0., -8.], [18., -2., 30.]]]).astype(dtype)
    out_shape = expect.shape
    output = np.full(out_shape, np.nan, dtype)
    return expect, input_np, output
