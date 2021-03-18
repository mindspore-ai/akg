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
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from akg.ops.math_gpu.sqrt import sqrt

def gen_data(shape, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    inputs = random_gaussian(
        shape, miu=1, sigma=0.1).astype(support_list[dtype])
    inputs = np.abs(inputs)
    expect = np.sqrt(inputs)
    output = np.full(expect.shape, np.nan, dtype)
    return output, expect, inputs


def test_ms_sqrt(shape, dtype, poly_sch=False):
    if poly_sch:
        mod = utils.op_build_test(sqrt, [shape], [
                             dtype], kernel_name="sqrt", attrs={"target": "cuda"})

    output, expect, inputs = gen_data(shape, dtype)
    output = utils.mod_launch(mod, (inputs, output), expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    inputs, expect = to_tvm_nd_array([inputs, expect])
    gpu_profiling(mod, inputs, expect, 400)
