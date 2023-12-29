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
from tests.common.test_op.ascend import gelu_grad
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def gelu_grad_data(shape, dtype):
    x = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    dy = np.random.rand(*shape).astype(dtype) * 4 - 2  # -2 ~ 2

    t = 0.044715
    s = 0.7978846
    tanh = np.tanh(s * (x + t * np.power(x, 3))).astype(dtype)

    res_grad = 0.5 * (1 + tanh + x * (1 - np.power(tanh, 2)) * (s * (1 + 3 * t * np.power(x, 2))))
    bench_mark = res_grad * dy
    output = np.full(shape, np.nan, dtype)

    return x, dy, bench_mark, output


def gelu_grad_execute(shape, dtype, attrs):
    np.random.seed(0)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = gelu_grad_compile(shape, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            x, dy, bench_mark, output = gelu_grad_data(shape, dtype)
            return mod, bench_mark, (x, dy, output)
        else:
            return mod
    else:
        mod = gelu_grad_compile(shape, dtype, attrs)
        x, dy, bench_mark, output = gelu_grad_data(shape, dtype)
        output = utils.mod_launch(mod, (x, dy, output), expect=bench_mark)

        rtol, atol = get_rtol_atol("gelu_grad", dtype)
        compare_res = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=False)

        return (x, dy), output, bench_mark, compare_res


def gelu_grad_compile(shape, dtype, attrs, kernel_name="gelu_grad", tuning=False):
    return utils.op_build_test(gelu_grad.gelu_grad, [shape, shape], [dtype, dtype], kernel_name=kernel_name,
                               attrs=attrs, tuning=tuning)
