# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
import math
import numpy as np
from functools import reduce
from akg import tvm
from akg.utils import kernel_exec as utils
from akg.ops.array import Reshape
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

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

def reshape_run(in_shape, out_shape, dtype, attrs):
    if attrs is None:
        attrs = {}
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = reshape_compile(in_shape, out_shape, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, in_shape, out_shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = reshape_compile(in_shape, out_shape, dtype, attrs)
        expect, input, output = gen_data(dtype, in_shape, out_shape)
        args = [input, output]
        if attrs.get("dynamic"):
            for index in range(len(out_shape) - 1):
                args.append(out_shape[index])
            for i in in_shape:
                args.append(i)
            block_dim = compute_blockdim(in_shape)
            args.append(block_dim)
        output = utils.mod_launch(mod, args, outputs=(1,), expect=expect)
        if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array(args, akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        rtol, atol = get_rtol_atol("reshape", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, in_shape, out_shape):
    input = np.random.randint(100, size=in_shape).astype(dtype)
    expect = np.reshape(input, out_shape)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input, output


def reshape_compile(in_shape, out_shape, dtype, attrs, kernel_name='reshape', tuning=False):
    if attrs.get("dynamic"):
        var_shape = []
        for i in range(len(in_shape)):
            var_shape.append(tvm.var("I" + str(i)))
        build_in_shape = var_shape
        total_size = reduce(lambda x, y: x * y, var_shape)
        out_var_shape = []
        if len(out_shape) >= 2:
            for i in range(len(out_shape) - 1):
                out_var_shape.append(tvm.var("O" + str(i)))
            out_size = reduce(lambda x, y: x * y, out_var_shape)
            out_var_shape.append(tvm.div(total_size, out_size))
        else:
            out_var_shape.append(total_size)
        build_out_shape = out_var_shape
    else:
        build_in_shape = in_shape
        build_out_shape = out_shape
    op_attr = [build_out_shape]
    return utils.op_build_test(Reshape, [build_in_shape], [dtype], op_attr, kernel_name=kernel_name, attrs=attrs, tuning=tuning)
