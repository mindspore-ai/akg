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
from akg import tvm
from akg.utils import kernel_exec as utils
from akg.ops.math import sum
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from akg.utils.result_analysis import np_bisect_sum
from akg.utils.dsl_create import get_reduce_out_shape
from tests.common.gen_random import random_gaussian
import math

def compute_blockdim(shape):
    size = 0
    if isinstance(shape, (list, tuple)):
        for i in shape:
            size = size * i
    elif isinstance(shape, int):
        size = shape
    else:
        size = 2
    return min(32, math.ceil(size / 8192 + 1))

def sum_execute(shape, reduce_axis, keepdims, dtype, attrs):
    if attrs is None:
        attrs = {}
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = sum_compile(shape, reduce_axis, keepdims, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input1, output = gen_data(dtype, keepdims, reduce_axis, shape)
            return mod, expect, (input1, output)
        else:
            return mod
    else:
        # op_attrs = [reduce_axis, keepdims]
        mod = sum_compile(shape, reduce_axis, keepdims, dtype, attrs)
        expect, input1, output = gen_data(dtype, keepdims, reduce_axis, shape)
        args = [input1, output]
        if attrs.get("dynamic"):
            for i in range(len(shape)):
                args.append(shape[i])
            block_dim = compute_blockdim(shape)
            args.append(block_dim)
        output = utils.mod_launch(mod, args, outputs=(1,), expect=expect)
        rtol, atol = get_rtol_atol("sum", dtype)
        return input1, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, keepdims, reduce_axis, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape, miu=1, sigma=0.1)
    input1 = input1.astype(support_list[dtype])
    if dtype == 'float16':
        expect = np_bisect_sum(input1, axis=reduce_axis, keepdims=keepdims)
    else:
        expect = np.sum(input1, axis=reduce_axis, keepdims=keepdims)
    out_shape = get_reduce_out_shape(shape, axis=reduce_axis, keepdims=keepdims)
    # if dump_data:
    #  with open('input1.bin', 'wb') as fo:
    #      fo.write(input1.astype(np.float16, copy=False))
    #  with open('output.bin', 'wb') as fo:
    #      fo.write(benchMark.astype(np.float16, copy=False))
    output = np.full(out_shape, np.nan, dtype)
    return expect, input1, output


def sum_compile(shape, reduce_axis, keepdims, dtype, attrs, kernel_name="sum", tuning=False):
    op_attrs = [reduce_axis, keepdims]
    if attrs is not None and attrs.get("dynamic"):
        var_shape = []
        for i in range(len(shape)):
            var_shape.append(tvm.var("I" + str(i)))
        attrs["enable_post_poly_loop_partition"] = False
        build_shape = var_shape
    else:
        build_shape = shape
    return utils.op_build_test(sum.sum_value, [build_shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=tuning)
