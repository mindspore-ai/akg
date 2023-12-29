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
from akg.ops.state.ascend import AssignAdd
from tests.common.gen_random import random_gaussian


def assign_add_run(input_shape, value_shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(AssignAdd, [input_shape, value_shape], [dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, value = gen_data(dtype, input_shape, value_shape)
            return mod, expect, {"args": (input, value), 'outputs': (0,), 'tuning': False}
        else:
            return mod
    else:
        mod = utils.op_build_test(AssignAdd, [input_shape, value_shape], [dtype, dtype],
                                  kernel_name='assign_add', attrs=attrs)
        expect, input, value = gen_data(dtype, input_shape, value_shape)
        result = utils.mod_launch(mod, (input, value), outputs=(0, ), expect=expect)
        return (value, input), result, expect, compare_tensor(result, expect, atol=5e-01, rtol=5e-03, equal_nan=True)


def gen_data(dtype, input_shape, value_shape):
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("tile_cce only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))
    input = random_gaussian(input_shape, miu=0.1, sigma=0.1)
    input = input.astype(support_list[dtype])
    value = random_gaussian(value_shape, miu=0.22, sigma=0.1)
    value = value.astype(support_list[dtype])
    expect = np.add(input, value)
    return expect, input, value
