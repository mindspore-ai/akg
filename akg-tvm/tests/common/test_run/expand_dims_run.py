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
from akg.ops.math import ExpandDims
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def expand_dims_run(shape, axis, dtype, kernel_name="expand_dims", attrs={}):
    op_attr = [axis]
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(ExpandDims, [shape], [dtype], op_attr, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(axis, dtype, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(ExpandDims, [shape], [dtype], op_attr, kernel_name=kernel_name, attrs=attrs)
        expect, input, output = gen_data(axis, dtype, shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)
        if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([input, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(axis, dtype, shape):
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    expect = np.expand_dims(input, axis=axis)
    res_shape = np.expand_dims(input, axis=axis).shape
    output = np.full(res_shape, np.nan, dtype)
    return expect, input, output
