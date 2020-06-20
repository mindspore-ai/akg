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

import numpy as np
from akg.utils import kernel_exec as utils
from test_op.bitwise_xor import bitwise_xor
from gen_random import random_gaussian


def bitwise_xor_run(x_shape, x_dtype, y_shape, y_dtype, attrs):
    shapes = [x_shape, y_shape]
    dtypes = [x_dtype, y_dtype]
    mod = utils.op_build_test(bitwise_xor, shapes, dtypes,
                              kernel_name="bitwise_xor", attrs=attrs)
    benchMark, inputs, output = gen_data(x_dtype, shapes)
    output = utils.mod_launch(mod, inputs + [output], expect=benchMark)
    return inputs, output, benchMark, np.array_equal(output, benchMark)


def gen_data(dtype, shapes):
    if len(shapes) != 2:
        raise RuntimeError("inputs num should be 2")
    x = random_gaussian(shapes[0]).astype(dtype)
    y = random_gaussian(shapes[1]).astype(dtype)
    benchMark = np.bitwise_xor(x, y)
    output = np.full(benchMark.shape, 0, dtype)
    return benchMark, [x, y], output
