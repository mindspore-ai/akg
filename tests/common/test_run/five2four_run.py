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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.array import five2four
from akg import tvm
from base import get_rtol_atol
from gen_random import random_gaussian
import math
def compute_blockdim(shape):
    size = 1
    if isinstance(shape, (list, tuple)):
        for i in shape:
            size = size * i
    elif isinstance(shape, int):
        size = shape
    else:
        size = 2
    return min(32, math.ceil(size / 16384))

def five2four_execute(shape4d, out_dtype, format, dtype, attrs):
    # Generate data
    op_attrs = [shape4d, out_dtype, format]
    if attrs is None:
        attrs = {}
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        input, bench_mark = gen_data(shape4d, dtype, out_dtype, format)
        shape_5d = input.shape
        mod = five2four_compile(shape_5d, dtype, op_attrs, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            output = np.full(shape4d, np.nan, out_dtype)
            return mod, bench_mark, (input, output)
        else:
            return mod
    else:
        input, bench_mark = gen_data(shape4d, dtype, out_dtype, format)
        # mod launch
        shape_5d = input.shape
        mod = five2four_compile(shape_5d, dtype, op_attrs, attrs)

        output = np.full(shape4d, np.nan, out_dtype)
        args = [input, output]
        # if attrs.get("dynamic"):
        #     for i in range(len(shape4d) - 1, -1, -1):
        #         args.append(shape4d[i])
        if attrs.get("dynamic"):
            args.append(shape_5d[0])
            args.append(shape_5d[1])
            args.append(shape_5d[4])
            block_dim = compute_blockdim(shape4d)
            args.append(block_dim)
        output = utils.mod_launch(mod, args, outputs=(1,), expect=bench_mark)
        # compare result
        rtol, atol = get_rtol_atol("five2four", dtype)
        compare_result = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=True)
        return input, output, bench_mark, compare_result


def five2four_compile(shape_5d, dtype, op_attrs, attrs, kernel_name='five2four', tuning=False):
    if attrs.get("dynamic"):
        var_shape = []
        shape4d, dst_type, _ = op_attrs
        channel_idx = 1
        for i in range(len(shape_5d)):
            if shape_5d[i] == 1:
                var_shape.append(shape_5d[i])
            else:
                var_shape.append(tvm.var("I" + str(i)))                
        build_shape = var_shape
    else:
        build_shape = shape_5d
    return utils.op_build_test(five2four.five2four, [build_shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=tuning)


def gen_data(shape, dtype, out_dtype, format):
    bench_mark = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    if format == 'NCHW':
        n, c, h, w = shape
        if c % 16 != 0:
            pad_input_shape = [n, c, h, w]
            pad_c = (c + 15) // 16 * 16
            pad_input_shape[1] = pad_c
            pad_input = np.zeros(pad_input_shape).astype(dtype)
            pad_input[:, :c, :, :] = bench_mark
            new_shape = [n, pad_c // 16, 16, h, w]
            input = pad_input.reshape(new_shape).transpose(0, 1, 3, 4, 2)
        else:
            new_shape = [n, c // 16, 16, h, w]
            input = bench_mark.reshape(new_shape).transpose(0, 1, 3, 4, 2)
    elif format == 'NHWC':
        n, h, w, c = shape
        if c % 16 != 0:
            pad_input_shape = [n, h, w, c]
            pad_c = (c + 15) // 16 * 16
            pad_input_shape[3] = pad_c
            pad_input = np.zeros(pad_input_shape).astype(dtype)
            pad_input[:, :, :, :c] = bench_mark
            new_shape = [n, h, w, pad_c // 16, 16]
            input = pad_input.reshape(new_shape).transpose(0, 3, 1, 2, 4)
        else:
            new_shape = [n, h, w, c // 16, 16]
            input = bench_mark.reshape(new_shape).transpose(0, 3, 1, 2, 4)
    bench_mark = bench_mark.astype(out_dtype)
    return input, bench_mark
