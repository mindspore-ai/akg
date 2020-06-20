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
from test_op.slice import slice as op_slice
from tensorio import compare_tensor
from gen_random import random_gaussian

def slice_run(shape, begin, size, dtype, attrs):
    op_attr = [begin, size]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(op_slice, [shape], [dtype], op_attr, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(begin, dtype, shape, size)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(op_slice, [shape], [dtype], op_attr, kernel_name='slice', attrs=attrs)
        expect, input, output = gen_data(begin, dtype, shape, size)
        output = utils.mod_launch(mod, (input, output), expect=expect)  # unified launch
        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(begin, dtype, shape, size):
    input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    out_shape = [size[i] if size[i] > 0 else shape[i] - begin[i] for i in range(len(shape))]
    slice_range = [slice(begin[i], begin[i] + out_shape[i]) for i in range(len(out_shape))]
    expect = input[tuple(slice_range)]
    output = np.full(out_shape, np.nan, dtype)
    return expect, input, output
