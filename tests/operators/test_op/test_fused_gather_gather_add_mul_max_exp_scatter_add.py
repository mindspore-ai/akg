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
from akg.ops.array_gpu.fused_gather_gather_add_mul_max_exp_scatter_add import fused_gather_gather_add_mul_max_exp_scatter_add
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_random import random_gaussian
import numpy as np
import akg
from copy import deepcopy

def gen_data(shape1, shape2, shape3, shape4, dtype1, dtype2, axis):
    input1 = random_gaussian(shape1).astype(dtype1)
    input2 = np.random.randint(low=0, high=shape1[axis], size=shape2).astype(dtype2)
    input3 = np.full(shape3, 0.2).astype(dtype1)
    input4 = np.random.randint(shape1[axis], size=shape4).astype(dtype2)

    gather_out1 = np.take(input1, input2, axis=axis)
    gather_out2 = np.take(input1, input2, axis=axis)
    add_out = np.add(gather_out1, gather_out2)
    mul_out = np.multiply(input3, add_out)
    max_out = np.maximum(add_out, mul_out)
    exp_out = np.exp(max_out)

    scatter_out = deepcopy(input1)
    np.add.at(scatter_out, input4, exp_out)

    return input1, input2, input3, input4, exp_out, scatter_out


def test_fused_gather_gather_add_mul_max_exp_scatter_add(input1_shape, input2_shape, input3_shape, input4_shape,
                                                         data_dtype, indices_type, axis, poly_sch=False, attrs=None):
    op_attrs = [axis]
    default_attrs = {"target": "cuda"}
    if attrs:
        default_attrs.update(attrs)
    mod = utils.op_build_test(fused_gather_gather_add_mul_max_exp_scatter_add,
                                  [input1_shape, input2_shape, input3_shape, input4_shape],
                                  [data_dtype, indices_type, data_dtype, indices_type],
                                  op_attrs=op_attrs, attrs=default_attrs, polyhedral=poly_sch,
                                  kernel_name="fused_gather_gather_add_mul_max_exp_scatter_add", )

    # gen data
    input1, input2, input3, input4, expect1, expect2 = gen_data(input1_shape, input2_shape, input3_shape, input4_shape,
                                                                data_dtype, indices_type, axis)

    output1 = np.zeros(expect1.shape, expect1.dtype)
    output2 = deepcopy(input1)
    output1, output2 = utils.mod_launch(mod, (input1, input2, input3, input4, output1, output2),
                                        outputs=(-2, -1))

    atol, rtol = get_rtol_atol("fused_gather_gather_add_mul_max_exp_scatter_add", data_dtype)
    res = compare_tensor(output1, expect1, rtol=rtol, atol=atol)
    res &= compare_tensor(output2, expect2, rtol=rtol, atol=atol)
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
        inputs = to_tvm_nd_array([input1, input2, input3, input4, output1, output2], akg.tvm.context(target_name, 0))
        target_profiling(mod, *inputs, target=target_name, repeat_time=attrs["repeat_time"])
