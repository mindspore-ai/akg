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
from akg.ops.math import logical_not



def logical_not_run(shape1, dtype, kernel_name, attrs_op=None, attrs=None):
    if attrs_op is not None:
        if attrs is not None:
            attrs.update(attrs_op)
        else:
            attrs = attrs_op
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(logical_not, [shape1], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input1, output = gen_data(shape1)
            return mod, expect, (input1, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(logical_not, [shape1], [dtype], kernel_name=kernel_name, attrs=attrs)
        expect, input1, output = gen_data(shape1)
        output = utils.mod_launch(mod, (input1, output), expect=expect)
        return input1, output, expect, np.array_equal(output, expect)


def gen_data(shape1):
    input1 = np.random.randint(2, size=shape1, dtype=np.bool)
    expect = np.logical_not(input1)
    output = np.full(expect.shape, False, dtype=np.bool)
    return expect, input1, output
