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
from akg.ops.math import Divide
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def div_run(shape1, shape2, dtype, attrs={}):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = div_compile(shape1, shape2, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input1, input2, output = gen_data(dtype, shape1, shape2)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        mod = div_compile(shape1, shape2, dtype, attrs)
        expect, input1, input2, output = gen_data(dtype, shape1, shape2)
        output = utils.mod_launch(mod, (input1, input2, output), expect=expect)
        rtol, atol = get_rtol_atol("div", dtype)
        if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([input1, input2, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        return (input1, input2), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)

def gen_data(dtype, shape1, shape2):
    input1 = random_gaussian(shape1, miu=10, sigma=1).astype(dtype)
    input2 = random_gaussian(shape2, miu=3, sigma=0.3).astype(dtype)
    if dtype == "int8" or dtype == "uint8" or dtype == "int32":
        expect = np.floor_divide(input1, input2)
    else:
        expect = np.true_divide(input1, input2)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input1, input2, output


def div_compile(shape1, shape2, dtype, attrs, kernel_name='div', tuning=False):
    return utils.op_build_test(Divide, [shape1, shape2], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
