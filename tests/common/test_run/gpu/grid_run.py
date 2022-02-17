# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.common.gen_random import random_gaussian
from tests.common.test_op.gpu.grid import grid_test


def gen_data(shape1, shape2, dtype1, dtype2):
    input1 = random_gaussian(shape1, miu=1, sigma=0.1).astype(dtype1)
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(dtype2)
    expect = grid_test(np.array(input1), np.array(input2))
    return input1, input2, expect


def grid_run(shape1, shape2, dtype1, dtype2, poly_sch=True, attrs=None):
    mod = utils.op_build_test(grid_test, [shape1, shape2],
                              [dtype1, dtype2], polyhedral=poly_sch,
                              attrs=attrs, kernel_name="grid")
    input1, input2, expect = gen_data(shape1, shape2, dtype1, dtype2)
    output = np.full(expect.shape, np.nan, expect.dtype)
    output = utils.mod_launch(mod, (input1, input2, output), expect=expect)
    rtol = atol = 1e-05
    res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Failed"))
    return (input1, input2), output, expect, res
