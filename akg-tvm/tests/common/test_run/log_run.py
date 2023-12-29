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
from akg.utils import kernel_exec as utils
from akg.ops.math import log
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def log_run(shape, dtype, kernel_name, attrs_op=None, attrs=None):
    input_shape = [shape]
    input_dtype = [dtype]
    if attrs_op is not None:
        if attrs is not None:
            attrs.update(attrs_op)
        else:
            attrs = attrs_op
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(log, input_shape, input_dtype, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input_, output = gen_data(dtype, shape)
            return mod, expect, (input_, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(log, input_shape, input_dtype, kernel_name=kernel_name, attrs=attrs)
        expect, input_, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (input_, output), expect=expect)
        rtol, atol = get_rtol_atol("log", dtype)
        if attrs.get("profiling", False):
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([input_, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        return input_, output, expect, compare_tensor(output, expect, rtol=rtol, atol = atol, equal_nan=True)

def gen_data(dtype, shape):
    input_ = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    input_ = np.abs(input_) + 1e-10
    expect = np.log(input_)
    output = np.full(shape, np.nan, dtype)
    return expect, input_, output
