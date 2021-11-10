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

"""
sqrt run define
"""

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import sigmoid_cross_entropy_with_logits
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def sigmoid_cross_entropy_with_logits_run(shape1, dtype1, shape2, dtype2, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(sigmoid_cross_entropy_with_logits.sigmoid_cross_entropy_with_logits,
                                  [shape1, shape2], [dtype1, dtype2], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, labels, logits, output = gen_data(dtype1, dtype2, shape1, shape2)
            return mod, expect, (labels, logits, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(sigmoid_cross_entropy_with_logits.sigmoid_cross_entropy_with_logits,
                                  [shape1, shape2], [dtype1, dtype2], kernel_name=kernel_name, attrs=attrs)
        expect, labels, logits, output = gen_data(dtype1, dtype2, shape1, shape2)
        output = utils.mod_launch(mod, (labels, logits, output), expect=expect)
        compare_res = compare_tensor(output, expect, rtol=5e-03, atol=5e-03, equal_nan=True)

        return (labels, logits), output, expect, compare_res


def gen_data(dtype1, dtype2, shape1, shape2):
    logits = random_gaussian(shape1, miu=0, sigma=1).astype(dtype1)
    labels = np.random.rand(*list(shape2)).astype(dtype2)  # Probabilities： 0~1
    relu_logits = np.maximum(logits, 0)
    neg_abs_logits = np.multiply(-1, np.abs(logits))
    ln_sigmoid_logits = np.log(1 + np.exp(neg_abs_logits))
    logits_mul_lables = np.multiply(logits, labels)
    expect = relu_logits - logits_mul_lables + ln_sigmoid_logits
    outShape = expect.shape
    output = np.full(outShape, np.nan, dtype1)
    return expect, labels, logits, output
