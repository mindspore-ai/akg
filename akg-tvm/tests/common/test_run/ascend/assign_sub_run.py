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
from tests.common.tensorio import compare_tensor
from akg.ops.math.ascend import assign_sub
from tests.common.gen_random import random_gaussian

def assign_sub_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(assign_sub, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs,
                                  tuning=t)
        if t:
            expect, input1, input2, output = gen_data(dtype, shape)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(assign_sub, [shape, shape], [dtype, dtype], kernel_name='assign_sub', attrs=attrs)
        expect, input1, input2, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (input1, input2, output), expect=expect)

        return input1, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape):
    """ Generate data for testing the op """
    input1 = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    input2 = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    expect = np.subtract(input1, input2)
    output = np.full(shape, np.nan, dtype)
    return expect, input1, input2, output
