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
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import discontinous_mov
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def discontinous_mov_run(shapes, dtype, attrs):
    # Result_Numpy
    shape1 = shapes[0]
    shape2 = shapes[1]
    op_attrs = [shape2]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(discontinous_mov.discontinous_mov, [shape1], [dtype], op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, exp_output, input = gen_data(dtype, shape1, shape2)
            return mod, exp_output, args
        else:
            return mod
    else:
        mod = utils.op_build_test(discontinous_mov.discontinous_mov, [shape1], [dtype], op_attrs,
                                  kernel_name='discontinous_mov', attrs=attrs)
        args, exp_output, input = gen_data(dtype, shape1, shape2)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)

        # compare result
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)
        return input, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shape1, shape2):
    support_list = {"float16": np.float16, "float32": np.float32}
    input = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    exp_output = input[0:len(input) + 1:2]
    exp_output = np.concatenate((exp_output, exp_output))
    exp_output = exp_output.reshape(shape2)
    # inputs and output to hold the data
    output = np.full(shape2, np.nan, dtype)
    args = [input, output]
    return args, exp_output, input
