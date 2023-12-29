# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.array.gpu import gather_nd
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from akg.utils.op_dsl import gather_nd_np
from akg.utils.gen_random import random_gaussian, gen_indices_gather_nd
import numpy as np
import akg


def gen_data(shape1, dtype1, shape2, dtype2):
    params = random_gaussian(shape1).astype(dtype1)
    out_dim1 = 1
    for i in range(len(shape2) - 1):
        out_dim1 = out_dim1 * shape2[i]

    indices = gen_indices_gather_nd(shape1, shape2, dtype2)
    expect = gather_nd_np(params, indices)

    return params, indices, expect

def gather_nd_run(shape1, dtype1, shape2, dtype2, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    mod = utils.op_build_test(gather_nd, [shape1, shape2], [dtype1, dtype2],
                 polyhedral=poly_sch, attrs=attrs, kernel_name="gather_nd")

    # gen data
    params, indices, expect = gen_data(shape1, dtype1, shape2, dtype2)
    output_shape = expect.shape

    if len(expect.shape) == 0:
        output_shape = (1, )
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (params, indices, output), expect = expect)

    atol, rtol = get_rtol_atol("gather_nd", dtype1)
    res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Failed"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs["profiling"]:
        params, indices, output = to_tvm_nd_array(
            [params, indices, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, params, indices, output, target=target_name, repeat_time=attrs["repeat_times"])
    return (params, indices), output, expect, res