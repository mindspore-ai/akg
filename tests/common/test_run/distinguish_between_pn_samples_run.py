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

import secrets
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import distinguish_between_pn_samples
from tests.common.gen_random import random_gaussian
secretsGenerator = secrets.SystemRandom()
def generate_expect(input, threshold):
    shape = input.shape
    if len(shape) != 3:
        raise RuntimeError(" the dims of input must be 3")
    if input.dtype == np.float32:
        input = input.astype(np.float16)
        threshold = np.array(threshold, dtype=np.float16)
    input = input - threshold
    output = np.full(shape[0:-1], shape[-1], 'int32')

    output = np.argmax(input, -1)

    for i in range(0, shape[0]):

        for j in range(0, shape[1]):

            if input[i, j, output[i, j]] < 0:
                output[i, j] = shape[2]
    output = output.astype('int32')
    return output


def distinguish_between_pn_samples_run(shape, threshold, dtype, attrs=None):
    if dtype == 'float32':
        threshold = int(threshold * 1000) / 1000.0

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(distinguish_between_pn_samples.distinguish_between_pn_samples, [shape], [dtype],
                                  op_attrs=[threshold], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, exp_output, input = gen_data(dtype, shape, threshold)
            return mod, exp_output, args
        else:
            return mod
    else:
        mod = utils.op_build_test(distinguish_between_pn_samples.distinguish_between_pn_samples, [shape], [dtype],
                                  op_attrs=[threshold], kernel_name="distinguish_between_pn_samples", attrs=attrs)
        args, exp_output, input = gen_data(dtype, shape, threshold)
        # run_testcase
        output = utils.mod_launch(mod, args, expect=exp_output)

        TestCase_Result = compare_tensor(output, exp_output, rtol=5e-03, equal_nan=True)
        return input, output, exp_output, TestCase_Result


def gen_data(dtype, shape, threshold):
    # Result_Numpy
    support_list = {"float16": np.float16, "float32": np.float32}
    input = random_gaussian(shape, miu=1 + secretsGenerator.randint(0, 9), sigma=0.1).astype(support_list[dtype])
    input = (input - input.min()) / (input.max() - input.min())
    if dtype == 'float32':
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    input[i, j, k] = ("%.3f" % input[i, j, k])
    input = input.astype(support_list[dtype])
    exp_output = generate_expect(input, threshold)
    output = np.full(exp_output.shape, np.nan, 'int32')
    args = (input, output)
    return args, exp_output, input
