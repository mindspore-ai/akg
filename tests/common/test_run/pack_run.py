# Copyright 2020 Huawei Technologies Co., Ltd
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

"""run function for pack"""

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op import pack
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol


def pack_run(shapes, dtype, axis, attrs):
    op_attrs = [axis]

    mod = utils.op_build_test(pack.pack,
                              [shapes], [dtype.lower()],
                              op_attrs, kernel_name="pack", attrs=attrs)
    inputs, expect, output = gen_data(axis, dtype, shapes)
    output = utils.mod_launch(mod, (*tuple(inputs), output), expect=expect)
    rtol, atol = get_rtol_atol("pack", dtype)
    return tuple(inputs), output, expect, \
        compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(axis, dtype, shapes):
    inputs = []
    for i in range(len(shapes)):
        shape = shapes[i]
        input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
        inputs.append(input)

    expect = np.concatenate(inputs, axis=axis)
    output_shape = shapes[0][:]
    if len(shapes) > 1:
        for i in range(1, len(shapes)):
            output_shape[axis] += shapes[i][axis]
    output = np.full(output_shape, np.nan, dtype)
    return inputs, expect, output
