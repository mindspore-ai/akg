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
from akg.ops.array_gpu import fused_gather_nd_reduce_sum_mul_unsorted_segment_sum
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_random import random_gaussian
import numpy as np
import akg
from copy import deepcopy

def gen_data(shape1, shape2, shape3, shape4, data_type, indices_type, axis, keepdims, num):
    input1 = random_gaussian(shape1).astype(data_type)
    out_dim1 = 1
    for i in range(len(shape2) - 1):
        out_dim1 = out_dim1 * shape2[i]
    input2 = np.zeros([shape2[-1], out_dim1]).astype(indices_type)
    for i in range(shape2[-1]):
        input2[i] = np.random.randint(low=0, high=shape1[i], size=out_dim1)
    input3 = random_gaussian(shape3).astype(data_type)
    prod = np.sum(input1[tuple(input2.tolist())], axis=axis, keepdims=keepdims) * input3

    input4 = np.random.randint(low=0, high=10, size=shape4).astype(indices_type)
    input5 = np.random.randint(low=0, high=10, size=shape4).astype(indices_type)
    expect1 = np.zeros((num,) + shape3[len(shape4):]).astype(data_type)
    expect2 = np.zeros((num,) + shape3[len(shape4):]).astype(data_type)
    np.add.at(expect1, input4, prod)
    np.add.at(expect2, input5, prod)

    input2 = input2.transpose()
    input2 = input2.reshape(shape2)
    return input1, input2, input3, input4, input5, expect1, expect2

def test_fused_gather_nd_reduce_sum_mul_unsorted_segment_sum(
    input1_shape, input2_shape, input3_shape, input4_shape, data_dtype, indices_type, axis, keepdims, num, poly_sch=True, attrs=None):
    op_attrs = [axis, keepdims, num]
    if not attrs:
        attrs = {"target": "cuda"}
    mod = utils.op_build_test(fused_gather_nd_reduce_sum_mul_unsorted_segment_sum.fused_gather_nd_reduce_sum_mul_unsorted_segment_sum,
                                                               [input1_shape, input2_shape, input3_shape, input4_shape, input4_shape], [data_dtype, indices_type, data_dtype, indices_type, indices_type], op_attrs=op_attrs,
                                                               attrs=attrs, kernel_name="fused_gather_nd_reduce_sum_mul_unsorted_segment_sum", polyhedral=poly_sch)
    input1, input2, input3, input4, input5, expect1, expect2 = gen_data(
        input1_shape, input2_shape, input3_shape, input4_shape, data_dtype, indices_type, axis, keepdims, num)

    output1 = np.zeros(expect1.shape, expect1.dtype)
    output2 = np.zeros(expect2.shape, expect2.dtype)
    output = utils.mod_launch(mod, (input1, input2, input3, input4, input5, output1, output2), outputs=(-2, -1), expect=(expect1, expect2))

    atol, rtol = get_rtol_atol("fused_gather_nd_reduce_sum_mul_unsorted_segment_sum", data_dtype)
    res = compare_tensor(output[0], expect1, rtol=rtol, atol=atol) and compare_tensor(output[1], expect2, rtol=rtol, atol=atol)
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
        input1, input2, input3, input4, input5, output1, output2 = to_tvm_nd_array(
            [input1, input2, input3, input4, input5, output1, output2], akg.tvm.context(target_name, 0))
        target_profiling(mod, input1, input2, input3, input4, input5, output1, output2, target=target_name, repeat_time=attrs["repeat_time"])
