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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import square_difference
from tests.common.gen_random import random_gaussian


def square_difference_run(shape1, shape2, dtype, kernel_name, attrs_op={}, cce_path="./", attrs={}):
    attrs.update(attrs_op)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(square_difference.square_difference, input_shapes=[shape1, shape2],
                                  input_types=[dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input1, input2, output = gen_data(dtype, shape1, shape2)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(square_difference.square_difference, input_shapes=[shape1, shape2],
                                  input_types=[dtype, dtype], kernel_name=kernel_name, attrs=attrs)
        expect, input1, input2, output = gen_data(dtype, shape1, shape2)
        source_code = mod.imported_modules[0].get_source()
        utils.create_code(kernel_name, cce_path, source_code)
        output = utils.mod_launch(mod, (input1, input2, output), expect=expect)
        return (input1, input2), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape1, shape2):
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = np.square(np.subtract(input1, input2))
    out_shape = expect.shape
    output = np.full(out_shape, np.nan, dtype)
    return expect, input1, input2, output
