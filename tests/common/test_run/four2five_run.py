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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.array import four2five
from akg import tvm
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
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
    return 32    
    # return min(32, math.ceil(size / 8192 + 1))

def four2five_execute(shape, dtype, format, dst_type, attrs=None):
    # Generate data
    op_attrs = [format, dst_type]
    if attrs is None:
        attrs = {}
    attrs["pragma_disable_whole_component"] = False
    attrs["pragma_disable_loop_reversal"] = False
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = four2five_compile(shape, dtype, op_attrs, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            output, input, bench_mark = gen_data(shape, dtype, format)
            return mod, bench_mark, (input, output)
        else:
            return mod
    else:
        mod = four2five_compile(shape, dtype, op_attrs, attrs)
        output, input, bench_mark = gen_data(shape, dtype, format, dst_type)

        args = [input,output]
        if attrs.get("dynamic"):
            if format == "NCHW":
                args.append(shape[0])
                args.append(shape[2])
                args.append(shape[3])
            elif format == "NHWC":
                args.append(shape[0])
                args.append(shape[1])
                args.append(shape[2])

            block_dim = compute_blockdim(shape)
            args.append(block_dim)
        output = utils.mod_launch(mod, args, outputs=(1,), expect=bench_mark)
        # compare result
        rtol, atol = get_rtol_atol("four2five", dtype)
        compare_result = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=True)
        return input, output, bench_mark, compare_result


def four2five_compile(shape, dtype, op_attrs, attrs, kernel_name='four2five', tuning=False):
    if attrs.get("dynamic"):
        var_shape = []
        format, dst_type = op_attrs
        channel_idx = 1
        if format == 'NCHW':
            channel_idx = 1
        elif format == 'NHWC':
            channel_idx = len(shape) - 1
        for i in range(len(shape)):
            if i == channel_idx:
                var_shape.append(shape[i])
            else:
                var_shape.append(tvm.var("I" + str(i)))
        build_shape = var_shape
    else:
        build_shape = shape
    return utils.op_build_test(four2five.four2five, [build_shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=tuning)



def gen_data(shape, dtype, format, dst_type):
    input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    if format == 'NCHW':
        n, c, h, w = shape
        if c % 16 != 0:
            pad_input_shape = [n, c, h, w]
            pad_c = (c + 15) // 16 * 16
            pad_input_shape[1] = pad_c
            pad_input = np.zeros(pad_input_shape).astype(dtype)
            pad_input[:, :c, :, :] = input
            new_shape = [n, pad_c // 16, h, w, 16]
            bench_mark = pad_input.reshape(n, pad_c // 16, 16, h, w).transpose(0, 1, 3, 4, 2).astype(dst_type)
        else:
            new_shape = [n, c // 16, h, w, 16]
            bench_mark = input.reshape(n, c // 16, 16, h, w).transpose(0, 1, 3, 4, 2).astype(dst_type)
    elif format == 'NHWC':
        n, h, w, c = shape
        if c % 16 != 0:
            pad_input_shape = [n, h, w, c]
            pad_c = (c + 15) // 16 * 16
            pad_input_shape[3] = pad_c
            pad_input = np.zeros(pad_input_shape).astype(dtype)
            pad_input[:, :, :, :c] = input
            new_shape = [n, pad_c // 16, h, w, 16]
            bench_mark = pad_input.reshape(n, h, w, pad_c // 16, 16).transpose(0, 3, 1, 2, 4).astype(dst_type)
        else:
            new_shape = [n, c // 16, h, w, 16]
            bench_mark = input.reshape(n, h, w, c // 16, 16).transpose(0, 3, 1, 2, 4).astype(dst_type)
    output = np.full(new_shape, np.nan, dst_type)
    return output, input, bench_mark
