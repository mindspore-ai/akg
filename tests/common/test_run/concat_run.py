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
from tests.common.test_op import concat
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def concat_run(shapes, dtype, axis, attrs):
    op_attrs = [axis]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(concat.concat, [shapes], [dtype.lower()], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, expect, inputs = gen_data(axis, dtype, shapes)
            return mod, expect, tuple(args)
        else:
            return mod
    else:
        mod = utils.op_build_test(concat.concat, [shapes], [dtype.lower()], op_attrs, kernel_name='concat', attrs=attrs)
        args, expect, inputs = gen_data(axis, dtype, shapes)
        output = utils.mod_launch(mod, tuple(args), expect=expect)
        return tuple(inputs), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(axis, dtype, shapes):
    inputs = []
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    for i in range(len(shapes)):
        shape = shapes[i]
        input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype.lower()])
        inputs.append(input)
    expect = np.concatenate(inputs, axis=axis)
    output_shape = shapes[0][:]
    if len(shapes) > 1:
        for i in range(1, len(shapes)):
            output_shape[axis] += shapes[i][axis]
    output = np.full(output_shape, np.nan, dtype)
    args = inputs
    args.append(output)
    return args, expect, inputs
