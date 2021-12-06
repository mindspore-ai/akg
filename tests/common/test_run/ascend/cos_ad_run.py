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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import cos_ad
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol

def cos_ad_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        expect, head_np, inputs = get_input_data(dtype, shape)
        mod = utils.op_build_test(cos_ad.cos_ad, [expect.shape, shape], [dtype, dtype], kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            # inputs and output to hold the data
            output = np.full(expect.shape, np.nan, dtype)
            return mod, expect, (head_np, inputs, output)
        else:
            return mod
    else:
        expect, head_np, inputs = get_input_data(dtype, shape)
        mod = utils.op_build_test(cos_ad.cos_ad, [expect.shape, shape], [dtype, dtype],
                                  kernel_name='cos_ad', attrs=attrs)
        # inputs and output to hold the data
        output = np.full(expect.shape, np.nan, dtype)
        output = utils.mod_launch(mod, (head_np, inputs, output), expect=expect)
        # compare result
        rtol, atol = get_rtol_atol("cos_ad", dtype)
        TestCase_Result = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=False)
        return (head_np, inputs), output, expect, TestCase_Result


def get_input_data(dtype, shape):
    # Generate data for testing the op
    inputs = random_gaussian(shape, miu=0, sigma=0.1).astype(dtype)
    head_np = random_gaussian(shape, miu=0, sigma=0.1).astype(dtype)
    expect = -np.sin(inputs) * head_np
    return expect, head_np, inputs
