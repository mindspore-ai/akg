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

from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
import numpy as np
from tests.common.gen_random import random_gaussian
from tests.common.test_op.sigmoid_ad import sigmoid_ad


def sigmoid_benchmark(input_np):
    neg_input = np.negative(input_np)
    exp_neg_input = np.exp(neg_input)
    exp_neg_input_1 = exp_neg_input + 1
    exp_neg_input_1_square = exp_neg_input_1 * exp_neg_input_1
    rec_input = np.reciprocal(exp_neg_input_1_square)
    sigmoid_grad = exp_neg_input * rec_input
    return sigmoid_grad


def sigmoid_ad_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(sigmoid_ad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, head_np, input_np, output = gen_data(dtype, shape)
            return mod, expect, (head_np, input_np, output)
        else:
            return mod
    else:
        expect, head_np, input_np, output = gen_data(dtype, shape)
        mod = utils.op_build_test(sigmoid_ad, [shape, shape], [dtype, dtype], kernel_name='sigmoid_ad', attrs=attrs)
        output = utils.mod_launch(mod, [head_np, input_np, output], expect=expect)
        return (head_np, input_np), output, expect, compare_tensor(output, expect, rtol=5e-02, atol=0.1)


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input_np = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    head_np = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    sigmoid_grad = sigmoid_benchmark(input_np)
    expect = sigmoid_grad * head_np
    output = np.full(expect.shape, np.nan, dtype)
    return expect, head_np, input_np, output
