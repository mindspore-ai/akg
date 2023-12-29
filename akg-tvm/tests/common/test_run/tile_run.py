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

from akg.utils import kernel_exec as utils
import numpy as np
from akg.ops.array import tile
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def tile_run(shape, dtype, multiples, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = tile_compile(shape, dtype, multiples, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            exp_output, inputs, output = gen_data(dtype, multiples, shape)
            return mod, exp_output, (inputs, output)
        else:
            return mod
    else:
        mod = tile_compile(shape, dtype, multiples, attrs)
        exp_output, inputs, output = gen_data(dtype, multiples, shape)
        acu_output = utils.mod_launch(mod, [inputs, output], expect=exp_output)
        if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([inputs, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        rtol, atol = get_rtol_atol("tile", dtype)
        return inputs, acu_output, exp_output, compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, multiples, shape):
    inputs = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    exp_output = np.tile(inputs, multiples)
    output = np.full(exp_output.shape, np.nan, dtype)
    return exp_output, inputs, output


def tile_compile(shape, dtype, multiples, attrs, kernel_name="tile", tuning=False):
    return utils.op_build_test(tile, [shape], [dtype], [multiples], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
