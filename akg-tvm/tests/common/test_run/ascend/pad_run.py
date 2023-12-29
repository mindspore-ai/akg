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

"""
sqrt run define
"""

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import pad
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def pad_run(shape, paddings, dtype, padtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        if len(paddings) == 0:
            ops_attrs = []
            mod = utils.op_build_test(pad.auto_pad, [shape], [dtype], ops_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        else:
            ops_attrs = [paddings, padtype]
            mod = utils.op_build_test(pad.pad, [shape], [dtype], ops_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, paddings, padtype, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        if len(paddings) == 0:
            ops_attrs = []
            mod = utils.op_build_test(pad.auto_pad, [shape], [dtype], ops_attrs, kernel_name=kernel_name, attrs=attrs)
        else:
            ops_attrs = [paddings, padtype]
            mod = utils.op_build_test(pad.pad, [shape], [dtype], ops_attrs, kernel_name=kernel_name, attrs=attrs)
        expect, input, output = gen_data(dtype, paddings, padtype, shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)

        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, paddings, padtype, shape):
    # Generate data for testing the op
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    if len(paddings) == 0:
        # auto pad to 16x
        pad_shape = [(x + 15) // 16 * 16 for x in shape]
        new_paddings = [[0, pad_shape[i] - shape[i]] for i in range(len(shape))]
        expect = np.pad(input, new_paddings, padtype)
    else:
        expect = np.pad(input, paddings, padtype)
    out_shape = expect.shape
    output = np.full(out_shape, np.nan, dtype)
    return expect, input, output
