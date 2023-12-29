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

"""expm1_run"""

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import expm1
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def expm1_run(shape, dtype, attrs):
    """expm1_run"""
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(expm1.expm1, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, inputs, output = gen_data(dtype, shape)
            return mod, expect, (inputs, output)
        return mod
    else:
        mod = utils.op_build_test(expm1.expm1, [shape], [dtype], kernel_name='expm1', attrs=attrs)
        expect, inputs, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (inputs, output), expect=expect)

        return inputs, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape):
    """gen_data"""
    inputs = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    expect = np.expm1(inputs)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, inputs, output
