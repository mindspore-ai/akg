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

"""rint_run"""
import numpy as np
from akg.utils import kernel_exec as utils
from test_op import rint
from tensorio import compare_tensor
from gen_random import random_gaussian
from base import get_rtol_atol

def rint_run(shape, dtype, attrs=None):
    """rint_run"""
    if attrs is None:
        attrs = {}

    mod = utils.op_build_test(rint.rint, [shape], [dtype], kernel_name='rint', attrs=attrs)
    args, exp_output, input_x = gen_data(shape, dtype)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("rint", dtype)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return input_x, acu_output, exp_output, testcase_result


def gen_data(shape, dtype):
    """gen_data"""
    input_x = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    exp_output = np.rint(input_x)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = [input_x, output]
    return args, exp_output, input_x
