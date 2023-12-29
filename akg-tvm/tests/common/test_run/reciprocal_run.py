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
import akg
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.math import reciprocal
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian, gen_epsilon
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def reciprocal_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = reciprocal_compile(shape, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input1, output = gen_data(dtype, shape)
            return mod, expect, (input1, output)
        else:
            return mod
    else:
        mod = reciprocal_compile(shape, dtype, attrs)
        expect, input1, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (input1, output), expect=expect)
        if attrs["profiling"]:
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([input1, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        rtol, atol = get_rtol_atol("reciprocal", dtype)
        return (input1,), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape, miu=1, sigma=0.1, epsilon=gen_epsilon(dtype)).astype(support_list[dtype])
    expect = np.reciprocal(input1)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input1, output


def reciprocal_compile(shape, dtype, attrs, kernel_name="reciprocal", tuning=False):
    return utils.op_build_test(reciprocal, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
