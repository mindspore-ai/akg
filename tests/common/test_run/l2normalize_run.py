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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import l2normalize


def l2normalize_run(shape, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(l2normalize.l2normalize, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, expect, input = gen_data(dtype, shape)
            return mod, expect, args
        else:
            return mod
    else:
        mod = utils.op_build_test(l2normalize.l2normalize, [shape], [dtype], kernel_name=kernel_name, attrs=attrs)
        args, expect, input = gen_data(dtype, shape)
        actual = utils.mod_launch(mod, args, expect=expect)

        if dtype == "float16":
            rtol = 1e-03
            atol = 1e-03
        else:
            rtol = 1e-04
            atol = 1e-04

        testcase_result = compare_tensor(actual, expect, rtol, atol, equal_nan=True)
        return input, actual, expect, testcase_result


def gen_data(dtype, shape):
    input = np.random.normal(size=shape).astype(dtype)
    if len(shape) > 1:
        reduce_sum = np.array((np.sum(np.square(input), axis=-1)) ** (0.5)).reshape((shape[0], 1))
        expect = input / reduce_sum
    else:
        reduce_sum = (np.sum(np.square(input), axis=-1)) ** (0.5)
        expect = input / reduce_sum
    output = np.full(expect.shape, np.nan, dtype)
    args = [input, output]
    return args, expect, input
