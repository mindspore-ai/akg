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
from akg.ops.math_gpu.log import log
from tests.common.gen_random import random_gaussian


def gen_data(in_shape, in_dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data = random_gaussian(in_shape, miu=1, sigma=0.1).astype(support_list[in_dtype])
    expect = np.log(data)
    output = np.full(expect.shape, np.nan, in_dtype)
    return data, output, expect


def test_ms_log(in_shape, in_dtype, poly_sch=False):
    if poly_sch:
        mod = utils.op_build_test(log, (in_shape,), (in_dtype,), kernel_name="log", attrs={"target": "cuda"})

    data, output, expect = gen_data(in_shape, in_dtype)
    args = (data, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-7)  # from 1e-8 changing to 1e-7
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
    return True
