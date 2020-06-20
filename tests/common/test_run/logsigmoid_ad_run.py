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
from test_op.logsigmoid_ad import logsigmoid_ad
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian


def logsigmoid_ad_benchmark(input_np):
    exp_input = np.exp(input_np)
    exp_input_1 = exp_input + 1
    logsigmoid_grad = np.reciprocal(exp_input_1)
    return logsigmoid_grad


def logsigmoid_ad_run(shape, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(logsigmoid_ad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, head_np, input_np, output = gen_data(dtype, shape)
            return mod, expect, (head_np, input_np, output)
        else:
            return mod

    else:
        expect, head_np, input_np, output = gen_data(dtype, shape)
        mod = utils.op_build_test(logsigmoid_ad, [shape, shape], [dtype, dtype], kernel_name="logsigmoid", attrs=attrs)
        output = utils.mod_launch(mod, [head_np, input_np, output], expect=expect)
        rtol, atol = get_rtol_atol("logsigmoid", dtype)
        return (head_np, input_np), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape):
    input_np = random_gaussian(shape, miu=1, sigma=0.5).astype(dtype)
    head_np = random_gaussian(shape, miu=1, sigma=0.5).astype(dtype)
    logsigmoid_grad = logsigmoid_ad_benchmark(input_np)
    expect = logsigmoid_grad * head_np
    output = np.full(expect.shape, np.nan, dtype)
    return expect, head_np, input_np, output