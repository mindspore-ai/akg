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
from tests.common.test_op.ascend.hpl_trsm import hpl_trsm


def gen_data(shape, dtype):
    num = shape[0]
    support_list = {"float16": np.float16, "float32": np.float32}

    one_tensor = np.ones((num, num))
    for i in range(num):
        for j in range(num):
            one_tensor[i, j] = min(min(i, j), 4) + 1
    input1 = np.tril(one_tensor).astype(support_list[dtype])
    expect = np.ones_like(input1)
    for i in range(num):
        for j in range(num):
            expect[i, j] = min(min(i, j), 6) + 1
    input2 = np.dot(input1, expect)
    output = np.full(expect.shape, np.nan, expect.dtype)
    return input1, input2, output, expect


def hpl_trsm_run(shape, dtype, poly_sch=True, attrs=None):

    attrs = {"enable_double_buffer": False,
             "enable_pre_poly_loop_partition": False,
             "enable_post_poly_loop_partition": False,
             "enable_to_three_address": False,
             "disable_align_dilation": True,
             "pragma_disable_schedule_shift": True,
             }

    mod = utils.op_build_test(hpl_trsm, [shape, shape], [dtype, dtype], kernel_name="hpl_trsm",
                              polyhedral=poly_sch, attrs=attrs)
    input1, input2, output, expect = gen_data(shape, dtype)
    output = utils.mod_launch(mod, (input1, input2, output), expect=expect, outputs=(1,))
    rtol = atol = 1e-04
    res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Failed"))
    return (input1, input2), output, expect, res
