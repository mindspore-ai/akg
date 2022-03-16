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
from akg.ops.array.gpu import TensorScatterAdd
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from akg.utils.op_dsl import tensor_scatter_add_np
from akg.utils.gen_random import gen_indices_tensor_scatter_add
import numpy as np
import akg


def gen_data(shape1, dtype1, shape2, dtype2):
    params = np.zeros(shape1, dtype1)
    update_shape = shape2[:-1] + shape1[shape2[-1]:]
    updates = np.random.random(update_shape).astype(dtype1)
    indices = gen_indices_tensor_scatter_add(shape1, shape2, dtype2)
    expect = tensor_scatter_add_np(params, indices, updates)
    return params, indices, updates, expect

def tensor_scatter_add_run(data_shape, data_type, indices_shape, indices_type, axis, poly_sch=True, attrs=None):
    op_attrs = [axis]
    default_attrs = {"target": "cuda"}
    if attrs:
        default_attrs.update(attrs)
    if len(indices_shape) > 1:
        updates_shape = indices_shape[:-1] + data_shape[indices_shape[-1]:]
    else:
        updates_shape = indices_shape + data_shape[1:]

    mod = utils.op_build_test(TensorScatterAdd, [data_shape, indices_shape, updates_shape], [data_type, indices_type, data_type],
                            attrs=default_attrs, kernel_name="tensor_scatter_add", polyhedral=poly_sch)

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
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs["profiling"]:
        params, indices, updates, output = to_tvm_nd_array(
            [params, indices, updates, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, params, indices, updates, output, target=target_name, repeat_time=attrs["repeat_times"])
    return (params, indices, updates), output, expect, res