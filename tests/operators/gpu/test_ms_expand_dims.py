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
from tests.common.test_run.expand_dims_run import gen_data
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from akg.ops.math_gpu.expand_dims import expand_dims

def test_expand_dims(shape1, axis, dtype, poly_sch=False):
    if poly_sch:
        mod = utils.op_build_test(expand_dims, [shape1], [dtype], op_attrs=[axis], attrs={"target": "cuda"}, kernel_name="expand_dims")    

    expect, input1, output = gen_data(axis, dtype, shape1)
    args = (input1, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
    input1, expect = to_tvm_nd_array([input1, expect])
    gpu_profiling(mod, input1, expect, 400)
        