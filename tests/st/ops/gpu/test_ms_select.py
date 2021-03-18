# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
# limitations under the License
import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.math_gpu.select import select
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor


def gen_data(shape_cond, shape_x, dtype_cond, dtype_x):
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int8": np.int8}
    cond = np.random.randint(0, 2, shape_cond).astype(support_list[dtype_cond])
    x1 = random_gaussian(shape_x, miu=1, sigma=0.1).astype(support_list[dtype_x])
    x2 = random_gaussian(shape_x, miu=1, sigma=0.1).astype(support_list[dtype_x])
    expect = np.where(cond, x1, x2)
    output = np.full(shape_x, np.nan, dtype_x)
    return expect, cond, x1, x2, output


def test_ms_select(shape_cond, shape_x, dtype_cond, dtype_x, poly_sch=False):
    if poly_sch:
        mod = utils.op_build_test(select, [shape_cond, shape_x, shape_x], [dtype_cond, dtype_x, dtype_x],
                                  kernel_name="select", attrs={"target": "cuda"})

    expect, cond, x1, x2, output = gen_data(shape_cond, shape_x, dtype_cond, dtype_x)
    output = utils.mod_launch(mod, (cond, x1, x2, output), expect=expect)
    res = compare_tensor(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
    return True
