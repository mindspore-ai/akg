# Copyright 2020 Huawei Technologies Co., Ltd
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

"""reverse_run"""
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import reverse
from tests.common.base import get_rtol_atol

def reverse_run(shape, dtype, axis, attrs=None):
    """reduce_any_d_run implementation"""
    if attrs is None:
        attrs = {}

    mod = utils.op_build_test(reverse.reverse, [shape], [dtype], kernel_name='reverse', op_attrs=[axis], attrs=attrs)
    args, exp_output, x = gen_data(dtype, shape, axis)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("reverse", dtype)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return x, acu_output, exp_output, testcase_result


def gen_data(dtype, shape, axis):
    # generate data for test
    if dtype == 'int32':
        low_bound = -1000
        high_bound = 1000
    else:
        low_bound = -1.0
        high_bound = 1.0

    input = np.random.uniform(low=low_bound, high=high_bound, size=tuple(shape)).astype(dtype)
    exp_output = np.flip(input, axis=axis)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = [input, output]
    return args, exp_output, input
