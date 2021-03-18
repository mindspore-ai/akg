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

from tests.common.tensorio import compare_tensor

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op import reduce_all

def reduce_all_run(shape, para_axis, keepdims=False, dtype="bool", kernel_name="reduce_all", attrs=None):
    input = np.full(shape, True, dtype=dtype)
    if len(shape) == 1:
        if shape[0] > 135 and np.random.randint(2, size=1):
            input[1 + 128] = False
            input[3 + 128] = False
            input[5 + 128] = False
    elif len(para_axis) == 1:
        if para_axis[0] == 1 and shape[0] >= 6:
            if shape[1] < 256:
                input[1][0] = False
                input[3][0] = False
                input[5][0] = False
            else:
                input[1][130] = False
                input[3][135] = False
                input[5][190] = False
        else:
            input[5][1] = False
            input[7][3] = False
            input[0][5] = False
    else:
        ind = (1,) * len(shape)
        input[ind] = False

    op_attrs = [para_axis, keepdims]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(reduce_all.reduce_all, [input.shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, output = gen_data(dtype, para_axis, keepdims, input)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        expect, output = gen_data(dtype, para_axis, keepdims, input)
        mod = utils.op_build_test(reduce_all.reduce_all, [input.shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs)
        output = utils.mod_launch(mod, (input, output), expect=expect)  # unified launch

        return input, output, expect, compare_tensor(output, expect, rtol=5e-02, atol=5e-04, equal_nan=True)


def gen_data(dtype, para_axis, keepdims, input):
    expect = np.all(input, axis=para_axis, keepdims=keepdims)
    shape = expect.shape
    if (len(shape) == 0):
        shape = (1,)
        expect = np.full(shape, expect, dtype)
    output = np.full(shape, np.nan, dtype)
    return expect, output
