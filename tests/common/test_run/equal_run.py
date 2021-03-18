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
from akg.ops.math import equal
from tests.common.gen_random import random_gaussian


def equal_run(shapes, dtype, kernel_name, attrs, cce_path="./"):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(equal.equal, shapes, [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            benchMark1, inputs1, output1 = gen_data(dtype, shapes)
            return mod, benchMark1, inputs1 + [output1]
        else:
            return mod
    else:
        mod = utils.op_build_test(equal.equal, shapes, [dtype, dtype], kernel_name=kernel_name, attrs=attrs)
        benchMark1, inputs1, output1 = gen_data(dtype, shapes)
        output1 = utils.mod_launch(mod, inputs1 + [output1], expect=benchMark1)

        # Also test the case where the inputs are equal
        if shapes[0] == shapes[1]:
            inputs2 = []
            inputs2.append(inputs1[0])
            inputs2.append(inputs1[0])
            benchMark2 = np.equal(inputs2[0], inputs2[1])
            output2 = np.full(benchMark2.shape, 0, bool)
            output2 = utils.mod_launch(mod, inputs2 + [output2], expect=benchMark1)
            testPass = (np.array_equal(output1, benchMark1) and np.array_equal(output2, benchMark2))
            return (inputs1, inputs2), (output1, output2), (benchMark1, benchMark2), testPass
        else:
            return inputs1, output1, benchMark1, np.array_equal(output1, benchMark1)


def gen_data(dtype, shapes):
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int8": np.int8, "uint8": np.uint8}
    inputs1 = []
    for i in range(len(shapes)):
        shape = shapes[i]
        input = random_gaussian(shape, miu=1, sigma=100).astype(support_list[dtype.lower()])
        inputs1.append(input)
    """
    inputs.append(np.ones(shapes[0], dtype=np.float16))
    for i in range(16):
        inputs[0][i]=0
    inputs.append(np.zeros(shapes[0], dtype=np.float16))
    """
    if len(inputs1) != 2:
        raise RuntimeError("inputs num should be 2")
    benchMark1 = np.equal(inputs1[0], inputs1[1])
    output1 = np.full(benchMark1.shape, 0, bool)
    return benchMark1, inputs1, output1
