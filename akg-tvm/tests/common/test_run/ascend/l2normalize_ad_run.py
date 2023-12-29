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
from tests.common.test_op.ascend import l2normalize_ad
from tests.common.gen_random import random_gaussian

def jacobian(head, x, reduce_sum, dtype):
    shape = x.shape
    jac = np.full(shape, np.nan, dtype)
    print(shape)
    if(len(shape) == 1):
        for i in range(shape[0]):
            sum = 0
            for k in range(shape[0]):
                res = x[i] * x[k]
                if k == i:
                    sum += head[i] * (reduce_sum - res)
                else:
                    sum += head[i] * (-res)
            jac[i] = sum
        return jac

    for i in range(shape[0]):
        for j in range(shape[1]):
            sum = 0
            for k in range(shape[1]):
                res = x[i][j] * x[i][k]
                if k == j:
                    sum += head[i][k] * (reduce_sum[i] - res)
                else:
                    sum += head[i][k] * (-res)
            jac[i][j] = sum
    return jac


def l2normalize_ad_run(shape, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(l2normalize_ad.l2normalize_ad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, expect, head, input = gen_data(dtype, shape)
            return mod, expect, args
        else:
            return mod
    else:
        mod = utils.op_build_test(l2normalize_ad.l2normalize_ad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs)
        args, expect, head, input = gen_data(dtype, shape)
        actual = utils.mod_launch(mod, args, expect=expect)
        testcase_result = compare_tensor(actual, expect, rtol=5e-03, atol=5e-03, equal_nan=True)
        return (input, head), actual, expect, testcase_result


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input = np.random.normal(size=shape).astype(dtype)
    head = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    head = np.abs(head)
    if len(shape) > 1:
        reduce_sum = np.sum(np.square(input), axis=-1)
        exp_sum = np.array(reduce_sum ** (1.5)).reshape((shape[0], 1))
        head_normlized = head / exp_sum
    else:
        reduce_sum = np.sum(np.square(input), axis=-1)
        exp_sum = reduce_sum ** (1.5)
        head_normlized = head / exp_sum
    expect = jacobian(head_normlized, input, reduce_sum, dtype)
    output = np.full(expect.shape, np.nan, dtype)
    args = [head, input, output]
    return args, expect, head, input
