# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import sliceeven
from tests.common.tensorio import compare_tensor


def sliceeven_run(shape, dtype, kernel_name="sliceeven_backward", attrs=None):
    kernel_name = utils.gen_name_kernel(kernel_name, dtype, shape)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(sliceeven.sliceeven, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input_, output = gen_data(dtype, shape)
            return mod, expect, (input_, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(sliceeven.sliceeven, [shape], [dtype], kernel_name=kernel_name, attrs=attrs)
        expect, input_, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (input_, output), expect=expect)
        return (input_,), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape):
    input_ = np.array(range(shape[0]))
    if (dtype == "int32"):
        input_ = input_.astype(np.int32)
    elif (dtype == "float16"):
        input_ = input_.astype(np.float16)
    elif (dtype == "float32"):
        input_ = input_.astype(np.float32)
    expect = np.where(input_ % 2 == 0, input_, 0)
    output = np.full(shape, 0.0, dtype)
    return expect, input_, output
