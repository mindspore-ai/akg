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
from akg.ops.math_gpu.log import log

def gen_data(in_shape, in_dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data = random_gaussian(in_shape, miu=1, sigma=0.1).astype(support_list[in_dtype])
    expect = np.log(data)
    output = np.full(expect.shape, np.nan, in_dtype)
    return data, output, expect

def test_ms_log(in_shape, in_dtype, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    mod = utils.op_build_test(log, (in_shape, ), (in_dtype, ), kernel_name="log", polyhedral=poly_sch, attrs=attrs)

    data, output, expect = gen_data(in_shape, in_dtype)
    args = (data, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-7) #  from 1e-8 changing to 1e-7
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
        args_list = to_tvm_nd_array(args, akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_time"])
