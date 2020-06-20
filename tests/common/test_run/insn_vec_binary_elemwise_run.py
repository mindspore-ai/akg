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
from test_op import insn_vec_binary_elemwise
from tensorio import compare_tensor
from gen_random import random_gaussian

def insn_vec_binary_elemwise_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(insn_vec_binary_elemwise.insn_vec_binary_elemwise, [shape, shape], [dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, exp_output, inputs = gen_data(dtype, shape)
            return mod, exp_output, args
        else:
            return mod
    else:
        mod = utils.op_build_test(insn_vec_binary_elemwise.insn_vec_binary_elemwise, [shape, shape], [dtype, dtype],
                                  kernel_name='insn_vec_binary_elemwise', attrs=attrs)
        args, exp_output, inputs = gen_data(dtype, shape)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)
        # compare result
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)

        return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shape):
    # Result_Numpy
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape, miu=0.5, sigma=0.01).astype(support_list[dtype])
    """ Generate the exp_output for validating the op """
    exp_output = input1 + input2
    # inputs and output to hold the data
    output = np.full(exp_output.shape, np.nan, dtype)
    inputs = [input1, input2]
    args = [input1, input2, output]
    return args, exp_output, inputs
