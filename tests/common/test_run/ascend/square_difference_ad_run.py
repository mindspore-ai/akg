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
from tests.common.test_op.ascend import square_difference_ad
from tests.common.gen_random import random_gaussian


def square_difference_ad_run(shape1, shape2, dtype, kernel_name, attrs_op={}, cce_path="./", attrs={}):
    attrs.update(attrs_op)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        expect, input1, input2, out_shape, support_list = gen_input_data(dtype, shape1, shape2)
        mod = utils.op_build_test(square_difference_ad.square_difference_ad, input_shapes=[out_shape, shape1, shape2],
                                  input_types=[dtype, dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, head_np, output = gen_data(dtype, expect, out_shape, support_list)
            return mod, expect, (head_np, input1, input2, output)
        else:
            return mod
    else:
        expect, input1, input2, out_shape, support_list = gen_input_data(dtype, shape1, shape2)
        expect, head_np, output = gen_data(dtype, expect, out_shape, support_list)
        mod = utils.op_build_test(square_difference_ad.square_difference_ad, input_shapes=[out_shape, shape1, shape2],
                                  input_types=[dtype, dtype, dtype], kernel_name='square_difference_ad', attrs=attrs)
        output = utils.mod_launch(mod, (head_np, input1, input2, output), expect=expect)

        return (head_np, input1, input2), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, expect, out_shape, support_list):
    head_np = random_gaussian(out_shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = expect * head_np
    output = np.full(out_shape, np.nan, dtype)
    return expect, head_np, output


def gen_input_data(dtype, shape1, shape2):
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = np.subtract(input1, input2) * 2
    out_shape = expect.shape
    return expect, input1, input2, out_shape, support_list
