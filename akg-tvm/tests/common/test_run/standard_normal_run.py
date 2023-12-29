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
import akg
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from akg.ops.array.gpu import standard_normal


def gen_data(shape):
    output = np.zeros(shape, dtype="float32")
    expect = np.zeros(shape, dtype="float32")
    return output, expect


def standard_normal_run(seed, shape, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    mod = utils.op_build_test(standard_normal, [], [], kernel_name="standard_normal", op_attrs=[seed, shape], attrs=attrs)

    output, expect = gen_data(shape)
    output = utils.mod_launch(mod, (output,), expect=expect)
    res = output.shape == expect.shape
    res &= abs(np.mean(output) - 0) < 1e-1
    res &= abs(np.std(output) - 1) < 1e-1
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
        output = to_tvm_nd_array(output, akg.tvm.context(target_name, 0))
        target_profiling(mod, output, target=target_name, repeat_time=attrs["repeat_times"])
    return output, output, expect, res