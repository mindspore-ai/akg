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
from tests.common.test_op.ascend import greater
from tests.common.gen_random import random_gaussian


def greater_run(shapes, dtype, kernel_name, attrs_op={}, cce_path="./", attrs={}):
    attrs.update(attrs_op)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(greater.greater, shapes, [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            benchMark, inputs, output = gen_data(shapes)
            return mod, benchMark, inputs + [output]
        else:
            return mod
    else:
        mod = utils.op_build_test(greater.greater, shapes, [dtype, dtype], kernel_name=kernel_name, attrs=attrs)
        benchMark, inputs, output = gen_data(shapes)
        output = utils.mod_launch(mod, inputs + [output], expect=benchMark)

        return inputs, output, benchMark, np.array_equal(output, benchMark)


def gen_data(shapes):
    inputs = []
    for i in range(len(shapes)):
        shape = shapes[i]
        input = random_gaussian(shape, miu=1, sigma=0.1).astype(np.float16)
        inputs.append(input)
    """
    inputs.append(np.ones(shapes[0], dtype=np.float16))
    for i in range(16):
        inputs[0][i]=0
    inputs.append(np.zeros(shapes[0], dtype=np.float16))
    """
    if len(inputs) != 2:
        raise RuntimeError("inputs num should be 2")
    benchMark = np.greater(inputs[0], inputs[1])
    output = np.full(benchMark.shape, 0, bool)
    return benchMark, inputs, output
