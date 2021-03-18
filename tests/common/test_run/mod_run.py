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
from tests.common.test_op import floormod
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def mod_run(shape, dtype, kernel_name, attrs, cce_path="./"):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(floormod.floormod, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input1, input2, output = gen_data(dtype, shape)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        expect, input1, input2, output = gen_data(dtype, shape)
        mod = floormod.floormod(shape, dtype, kernel_name, attrs)
        output = utils.mod_launch(mod, (input1, input2, output), expect=expect)  # unified launch
        return (input1, input2), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    input1 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape, miu=2, sigma=0.1).astype(support_list[dtype])
    expect = np.mod(input1, input2)
    output = np.full(shape, np.nan, dtype)
    return expect, input1, input2, output
