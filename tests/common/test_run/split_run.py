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
from test_op import split
from tensorio import compare_tensor
from gen_random import random_gaussian

def split_run(shape, num_or_size_splits, split_axis, dtype, attrs):
    op_attrs = [num_or_size_splits, split_axis]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(split.split, [shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            arg1, arg2, expect, input = gen_data(dtype, num_or_size_splits, shape, split_axis)
            return mod, expect, arg1, arg2
        else:
            return mod
    else:
        arg1, arg2, expect, input = gen_data(dtype, num_or_size_splits, shape, split_axis)
        mod = utils.op_build_test(split.split, [shape], [dtype], op_attrs, kernel_name='split', attrs=attrs)
        output = utils.mod_launch(mod, arg1, arg2, expect=expect)

        if num_or_size_splits == 1 or (isinstance(num_or_size_splits, (list, tuple)) and len(num_or_size_splits) == 1):
            TestCase_Result = compare_tensor(output, expect, rtol=5e-1, atol=5e-3, equal_nan=True)
        else:
            TestCase_Result = all(map(lambda x, y: compare_tensor(x, y, rtol=5e-1, atol=5e-3), output, expect))
        return input, output, expect, TestCase_Result


def gen_data(dtype, num_or_size_splits, shape, split_axis):
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    # num_or_size_splits can be list or a num
    if isinstance(num_or_size_splits, (list, tuple)):
        if len(num_or_size_splits) == 1:
            expect = [input]
        else:
            size_splits = [num_or_size_splits[0]]
            for i in range(len(num_or_size_splits) - 2):
                size_splits.append(num_or_size_splits[i + 1] + size_splits[i])
            expect = np.split(input, size_splits, split_axis)
    else:
        expect = np.split(input, num_or_size_splits, split_axis)

    # use expect shape for output init
    output = []
    arg1 = [input]
    for item in expect:
        out_item = np.full(item.shape, np.nan, dtype)
        output.append(out_item)
        arg1.append(out_item)
    arg2 = []
    for i in range(len(expect)):
        arg2.append(i - len(expect))
    return arg1, arg2, expect, input
