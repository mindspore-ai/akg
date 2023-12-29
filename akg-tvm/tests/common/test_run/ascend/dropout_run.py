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
import secrets
import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import dropout
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol

secretsGenerator = secrets.SystemRandom()


def _get_mask_len(shape_tensor):
    mask_length = 1
    for x in shape_tensor:
        mask_length *= x
    if mask_length % 8 != 0:
        mask_length = (mask_length + 7) // 8 * 8
    return mask_length


def gen_data(dtype, shape_tensor, keep_prob):
    mask_length = _get_mask_len(shape_tensor)
    real_length = 1
    for x in shape_tensor:
        real_length *= x

    # Generate data for testing the op
    input = np.random.randint(-5, 6, shape_tensor).astype(dtype)

    mask_input = np.zeros([mask_length // 8]).astype("uint8")
    mask_tensor = np.zeros([mask_length]).astype(dtype)

    for i in range(mask_length // 8):
        mask = 0
        for j in range(8):
            rst = secretsGenerator.randint(0, 65535)
            if rst <= 65535 * keep_prob:
                mask_tensor[i * 8 + j] = 1
                mask |= 1 << j
        mask_input[i] = mask

    mask_tensor = mask_tensor[0:real_length]
    mask_tensor = np.reshape(mask_tensor, shape_tensor)
    coef = np.zeros([1]).astype(dtype)
    coef[0] = (1.0 / keep_prob)
    expect = input * mask_tensor * coef[0]
    expect = expect.astype("float16").astype(dtype)
    output = np.full(shape_tensor, np.nan, dtype)

    return expect, input, output, mask_input


def dropout_execute(shape_tensor, keep_prob, dtype, kernel_name, attrs=None):
    # Create op
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = dropout_compile(shape_tensor, keep_prob, dtype, kernel_name, attrs, tuning=t)
        if t:
            expect, input, output, mask = gen_data(dtype, shape_tensor, keep_prob)
            return mod, expect, (input, mask, output)
        else:
            return mod
    else:
        mod = dropout_compile(shape_tensor, keep_prob, dtype, kernel_name, attrs)
        expect, input, output, mask = gen_data(dtype, shape_tensor, keep_prob)
        output = utils.mod_launch(mod, (input, mask, output), expect=expect)

        source_code = mod.imported_modules[0].get_source()
        utils.create_code(kernel_name, "./", source_code)

        rtol, atol = get_rtol_atol("dropout", dtype)
        return (input, mask), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def dropout_compile(shape_tensor, keep_prob, dtype, kernel_name, attrs=None, tuning=False):
    shape_list = [shape_tensor, [_get_mask_len(shape_tensor) // 8]]
    return utils.op_build_test(dropout.dropout_do_mask, shape_list, [dtype, 'uint8'],
                               [keep_prob], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
