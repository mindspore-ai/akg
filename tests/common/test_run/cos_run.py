# Copyright 2019 Huawei Technologies Co., Ltd
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

"""cos run function."""
import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import cos
from base import get_rtol_atol
from gen_random import random_gaussian

def cos_run(shape, dtype, attrs):
    # Generate data for testing the op
    inputs = random_gaussian(shape, miu=0, sigma=0.1).astype(dtype)
    expect = np.cos(inputs)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(cos.cos, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            return mod, expect, (inputs, output)
        else:
            return mod

    else:
        mod = utils.op_build_test(cos.cos, [shape], [dtype], kernel_name='cos', attrs=attrs)

    # result_tvm
    output = utils.mod_launch(mod, (inputs, output))

    # compare result
    rtol, atol = get_rtol_atol("cos", dtype)
    TestCase_Result = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=False)

    return inputs, output, expect, TestCase_Result