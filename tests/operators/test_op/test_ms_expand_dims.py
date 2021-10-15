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
from tests.common.test_run.expand_dims_run import gen_data
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from akg.ops.math_gpu.expand_dims import expand_dims

def test_expand_dims(shape1, axis, dtype, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    mod = utils.op_build_test(expand_dims, [shape1], [dtype], op_attrs=[axis], polyhedral=poly_sch, attrs=attrs, kernel_name="expand_dims")

    expect, input1, output = gen_data(axis, dtype, shape1)
    args = (input1, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
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
        args_list = to_tvm_nd_array([input1, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_time"])
