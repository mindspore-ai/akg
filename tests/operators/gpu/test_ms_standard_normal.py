# Copyright 2021 Huawei Technologies Co., Ltd
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
from akg.ops.array_gpu.standard_normal import standard_normal

def gen_data(shape):
    output = np.zeros(shape, dtype="float32")
    expect = np.zeros(shape, dtype="float32")
    return output, expect

def test_ms_standard_normal(seed, shape, poly_sch=False):
    if poly_sch:
        mod = utils.op_build_test(standard_normal,
                                  [],[],
                                  kernel_name="StandardNormal",
                                  op_attrs=[seed, shape],
                                  attrs={"target": "cuda"})

    output, expect = gen_data(shape)
    output = utils.mod_launch(mod, (output,), expect=expect)
    res = output.shape == expect.shape
    res &= abs(np.mean(output) - 0) < 1e-1
    res &= abs(np.std(output) - 1) < 1e-1
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
