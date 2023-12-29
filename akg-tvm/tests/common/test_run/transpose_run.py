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
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.array import transpose
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def transpose_data(shape, axes, dtype):
    data_input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    bench_mark = data_input.transpose(axes)
    return data_input, bench_mark

def transpose_run(shape, axes, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = transpose_compile(shape, axes, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            bench_mark, data_input, output = gen_data(axes, dtype, shape)
            return mod, bench_mark, (data_input, output)
        else:
            return mod
    else:
        mod = transpose_compile(shape, axes, dtype, attrs)
        bench_mark, data_input, output = gen_data(axes, dtype, shape)
        output = utils.mod_launch(mod, (data_input, output), expect=bench_mark)
        if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([data_input, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        # compare result
        rtol, atol = get_rtol_atol("transpose", dtype)
        compare_result = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=True)
        return data_input, output, bench_mark, compare_result


def gen_data(axes, dtype, shape):
    # Generate data
    data_input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    bench_mark = data_input.transpose(axes)
    # mod launch
    out_shape = ()
    for i in axes:
        out_shape = out_shape + (shape[i],)
    output = np.full(out_shape, np.nan, dtype)
    return bench_mark, data_input, output


def transpose_compile(shape, axes, dtype, attrs, kernel_name='transpose', tuning=False):
    op_attrs = [axes]
    return utils.op_build_test(transpose, [shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=tuning)