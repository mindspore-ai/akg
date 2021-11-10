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
from tests.common.test_op.ascend import rsqrt_grad
from tests.common.gen_random import random_gaussian


def rsqrt_grad_run(shape, dtype, kernel_name, attrs_op={}, cce_path="./", attrs={}):
    attrs.update(attrs_op)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(rsqrt_grad.rsqrt_grad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, grad, input, output = gen_data(dtype, shape)
            return mod, expect, (input, grad, output)
        else:
            return mod
    else:
        expect, grad, input, output = gen_data(dtype, shape)
        mod = utils.op_build_test(rsqrt_grad.rsqrt_grad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs)
        output = utils.mod_launch(mod, (input, grad, output), expect=expect)

        return (input, grad), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    grad = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = np.power(input, 3) * (-0.5) * grad
    output = np.full(shape, np.nan, dtype)
    return expect, grad, input, output
