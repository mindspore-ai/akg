# Copyright 2022 Huawei Technologies Co., Ltd
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
# limitations under the License.
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend.hpl_lu import hpl_lu


def gen_data(shape, dtype):
    num = shape[0]
    support_list = {"float16": np.float16, "float32": np.float32}

    one_tensor = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            one_tensor[i, j] = min(min(i, j), 4) + 1
    upper_matrix = np.triu(one_tensor).astype(support_list[dtype])
    lower_matrix = np.tril(one_tensor).astype(support_list[dtype])
    for i in range(num):
        lower_matrix[i, i] = 1.0
    input1 = np.dot(lower_matrix, upper_matrix)
    expect = upper_matrix + lower_matrix - np.eye(num)
    output = np.full(expect.shape, np.nan, expect.dtype)
    return input1, output, expect


def hpl_lu_run(shape, dtype, poly_sch=True, attrs=None):

    attrs = {
        "enable_double_buffer": False,
        "enable_pre_poly_loop_partition": False,
        "enable_post_poly_loop_partition": False,
    }

    mod = utils.op_build_test(hpl_lu, [shape, ], [dtype, ], kernel_name="hpl_lu",
                              polyhedral=poly_sch, attrs=attrs)
    input1, output, expect = gen_data(shape, dtype)
    output = utils.mod_launch(mod, (input1, output), expect=expect, outputs=(0,))
    rtol = atol = 1e-04
    res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Failed"))
    return (input1,), output, expect, res
