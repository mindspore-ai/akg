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

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op.elemwise_mul_ad import elemwise_mul_ad
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def elemwise_mul_ad_run(shape, dtype, attrs, polyhedral=True):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(elemwise_mul_ad, [shape, shape, shape], [dtype, dtype, dtype],
                                  kernel_name='elemwise_mul_ad', attrs=attrs, tuning=t)
        if t:
            expect, head_np, input1, input2, output = gen_data(dtype, shape)
            return mod, expect, (head_np, input1, input2, output)
        else:
            return mod
    else:
        expect, head_np, input1, input2, output = gen_data(dtype, shape)
        if polyhedral:
            mod = utils.op_build_test(elemwise_mul_ad, [shape, shape, shape], [dtype, dtype, dtype],
                                  kernel_name='elemwise_mul_ad', polyhedral=polyhedral, attrs=attrs)
            output = utils.mod_launch(mod, (head_np, input1, input2, output), expect=expect)

        return (head_np, input1, input2), output, expect, compare_tensor(output, expect, rtol=5e-03, atol=5e-03, equal_nan=True)


def gen_data(dtype, shape):
    input1 = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    input2 = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    head_np = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    expect = head_np*input2
    output = np.full(shape, np.nan, dtype)
    return expect, head_np, input1, input2, output
