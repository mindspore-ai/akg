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
from akg.ops.math.ascend import Sign
from tests.common.gen_random import random_gaussian

def sign_run(shape, dtype, attrs):
    in_shape = [shape]
    in_dtype = [dtype]
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(Sign, in_shape, in_dtype, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        expect, input, output = gen_data(dtype, shape)
        mod = utils.op_build_test(Sign, in_shape, in_dtype, kernel_name='sign', attrs=attrs)
        output = utils.mod_launch(mod, (input, output), expect=expect)
        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape):
    input = random_gaussian(shape, miu=1, sigma=10).astype(dtype)
    expect = np.sign(input)
    output = np.full(shape, np.nan, expect.dtype)
    return expect, input, output
