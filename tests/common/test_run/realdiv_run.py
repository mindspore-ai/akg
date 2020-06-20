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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.math import realdiv
from akg.utils.dsl_create import produce_shapes
from gen_random import random_gaussian

def gen_expect(input1, input2):
    a, b, out_shape = produce_shapes(input1.shape, input2.shape)
    n_input1 = np.broadcast_to(input1, out_shape)
    n_input2 = np.broadcast_to(input2, out_shape)

    sign2 = np.sign(n_input2)
    input2 = np.add(np.abs(n_input2), 1)
    input2 = np.multiply(n_input2, sign2)
    expect = np.divide(n_input1, n_input2)
    return expect


def realdiv_run(shape, shape2, dtype, kernel_name, attrs, cce_path="./"):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(realdiv.realdiv, [shape, shape2], [dtype, dtype], kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            expect, input1, input2, output = gen_data(dtype, shape, shape2)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        expect, input1, input2, output = gen_data(dtype, shape, shape2)
        mod = utils.op_build_test(realdiv.realdiv, [shape, shape2], [dtype, dtype], kernel_name=kernel_name,
                                  attrs=attrs)
        output = utils.mod_launch(mod, (input1, input2, output), expect=expect)

        return (input1, input2), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape, shape2):
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = gen_expect(input1, input2)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input1, input2, output
