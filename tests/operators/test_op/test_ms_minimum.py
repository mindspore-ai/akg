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
import akg
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.tensorio import compare_tensor
from akg.ops.math_gpu.minimum import minimum

def gen_data(shape1, shape2, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhs = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    rhs = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = np.minimum(lhs, rhs)
    output = np.full(expect.shape, np.nan, expect.dtype)
    return lhs, rhs, output, expect

def test_ms_minimum(shape1, shape2, dtype, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    mod = utils.op_build_test(minimum, (shape1, shape2), (dtype, dtype), kernel_name="minimum",
                        polyhedral=poly_sch, attrs=attrs)

    lhs, rhs, output, expect = gen_data(shape1, shape2, dtype)
    args = (lhs, rhs, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = compare_tensor(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs["profiling"]:
        args_list = to_tvm_nd_array([lhs, rhs, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_time"])
