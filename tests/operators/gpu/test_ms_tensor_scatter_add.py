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

from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.array_gpu import tensor_scatter_add
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_random import random_gaussian, gen_indices_tensor_scatter_add
from tests.common.test_utils import tensor_scatter_add_np
import numpy as np


def gen_data(shape1, dtype1, shape2, dtype2):
    params = np.zeros(shape1, dtype1)
    update_shape = shape2[:-1] + shape1[shape2[-1]:]
    updates = np.random.random(update_shape).astype(dtype1)
    indices = gen_indices_tensor_scatter_add(shape1, shape2, dtype2)
    expect = tensor_scatter_add_np(params, indices, updates)
    return params, indices, updates, expect

def test_ms_tensor_scatter_add(data_shape, data_type, indices_shape, indices_type, axis, poly_sch=False, attrs=None):
    op_attrs = [axis]
    default_attrs = {"target": "cuda"}
    if attrs:
        default_attrs.update(attrs)
    if len(indices_shape) > 1:
        updates_shape = indices_shape[:-1] + data_shape[indices_shape[-1]:]
    else:
        updates_shape = indices_shape + data_shape[1:]

    if poly_sch:
        mod = utils.op_build_test(tensor_scatter_add.tensor_scatter_add,
                                  [data_shape, indices_shape, updates_shape], [data_type, indices_type, data_type],
                                   attrs=default_attrs, kernel_name="tensor_scatter_add", )

    # gen data
    indices_shape = indices_shape + (1,) if len(indices_shape) == 1 else indices_shape
    params, indices, updates, expect = gen_data(data_shape, data_type, indices_shape, indices_type)
    output_shape = expect.shape

    if len(expect.shape) == 0:
        output_shape = (1, )
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (params, indices, updates, output), expect = expect)

    atol, rtol = get_rtol_atol("tensor_scatter_add", data_type)
    res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Failed"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    params, indices, updates, output, expect = to_tvm_nd_array(
        [params, indices, updates, output, expect])
    gpu_profiling(mod, params, indices, updates, output, expect, repeat_time=400)
