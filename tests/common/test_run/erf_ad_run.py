# Copyright 2019 Huawei Technologies Co., Ltd
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
from tests.common.test_op.erf_ad import erf_ad
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def erf_diff(x):
    """compute gradient of x by `2/sqrt(pi) * e^{-2x^2}`"""
    return 1.12837916709551257 * np.exp(-np.square(x))

def erf_ad_run(shape, dtype, kernel_name, attrs, cce_path="./"):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(erf_ad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, head_np, input_np, output = gen_data(dtype, shape)
            return mod, expect, (input_np, head_np, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(erf_ad, [shape, shape], [dtype, dtype], kernel_name="erf", attrs=attrs)
        expect, head_np, input_np, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (head_np, input_np, output), expect=expect)
        return input_np, output, expect, compare_tensor(output, expect, rtol=5e-02, atol=0.05)


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input_np = random_gaussian(shape, miu=0, sigma=0.1).astype(support_list[dtype])
    idx = np.where(np.logical_and(input_np < 0.5, input_np >= 0))
    input_np[idx] = input_np[idx] + 0.5
    idx = np.where(np.logical_and(input_np > -0.5, input_np < 0))
    input_np[idx] = input_np[idx] - 0.5
    expect = erf_diff(input_np)
    head_np = random_gaussian(shape, miu=0, sigma=0.1).astype(support_list[dtype])
    expect = expect * head_np
    output = np.full(shape, 1, support_list[dtype])
    return expect, head_np, input_np, output
