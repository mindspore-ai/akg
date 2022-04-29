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
from akg.utils import kernel_exec as utils
from akg.ops.math import less_equal
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def less_equal_run(shapes, dtype, kernel_name="less_equal", attrs_op=None, attrs=None):
    if attrs_op is not None:
        if attrs is not None:
            attrs.update(attrs_op)
        else:
            attrs = attrs_op
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = less_equal_compile(shapes, dtype, kernel_name, attrs, tuning=t)
        if t:
            expect, inputs, output = gen_data(dtype, shapes)
            return mod, expect, (inputs + output)
        else:
            return mod
    else:
        mod = less_equal_compile(shapes, dtype, kernel_name, attrs)
        expect, inputs, output = gen_data(dtype, shapes)
        output = utils.mod_launch(mod, inputs + [output], expect=expect)
        if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([inputs, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        rtol, atol = get_rtol_atol("less_equal", dtype)
        return inputs, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)

def gen_data(dtype, shapes):
    inputs = []
    for i in range(len(shapes)):
        shape = shapes[i]
        input_ = random_gaussian(shape, miu=1, sigma=(0.1 + 100 * i)).astype(dtype)
        inputs.append(input_)
    if len(inputs) != 2:
        raise RuntimeError("inputs num should be 2")
    expect = np.less_equal(inputs[0], inputs[1])
    output = np.full(expect.shape, np.nan, "bool")
    return expect, inputs, output


def less_equal_compile(shapes, dtype, kernel_name, attrs, tuning=False):
    return utils.op_build_test(less_equal, shapes, [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
