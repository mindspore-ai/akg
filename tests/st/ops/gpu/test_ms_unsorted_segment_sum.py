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
from copy import deepcopy
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.array_gpu import unsorted_segment_sum
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_random import random_gaussian, gen_indices_unsorted_segment_sum


def gen_data(shape1, dtype1, shape2, dtype2, num):
    input1 = random_gaussian(shape1).astype(dtype1)
    input2 = gen_indices_unsorted_segment_sum(shape1, shape2, dtype2, num)
    expect = np.zeros((num,) + shape1[len(shape2):]).astype(dtype1)
    np.add.at(expect, input2, input1)
    return input1, input2, expect

def test_ms_unsorted_segment_sum(data_shape, data_type, indices_shape, indices_type, num, poly_sch=False, attrs=None):
    op_attrs = [num]
    default_attrs = {"target": "cuda"}
    if attrs:
        default_attrs.update(attrs)

    if poly_sch:
        mod = utils.op_build_test(unsorted_segment_sum.unsorted_segment_sum,
                                  [data_shape, indices_shape], [data_type, indices_type, data_type], op_attrs=op_attrs,
                                   attrs=default_attrs, kernel_name="unsorted_segment_sum", )

    # gen data
    input1, input2, expect = gen_data(data_shape, data_type, indices_shape, indices_type, num)
    output_shape = expect.shape

    if len(expect.shape) == 0:
        output_shape = (1, )
    #output = np.full(output_shape, np.nan, expect.dtype)
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (input1, input2, output), expect = expect)

    atol, rtol = get_rtol_atol("unsorted_segment_sum", data_type)
    res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Failed"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    input1, input2, output, expect = to_tvm_nd_array(
        [input1, input2, output, expect])
    gpu_profiling(mod, input1, input2, output, expect, repeat_time=400)
