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
from test_op import sigmoid
from tensorio import compare_tensor
from gen_random import random_gaussian

def sigmoid_run(shape, dtype, kernel_name, attrs):
    input_shape = [shape]
    input_dtype = [dtype]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(sigmoid.sigmoid, input_shape, input_dtype, attrs=attrs,
                                  kernel_name=kernel_name,
                                  tuning=t)
        if t:
            expect, input, output = gen_data(dtype, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        expect, input, output = gen_data(dtype, shape)
        mod = utils.op_build_test(sigmoid.sigmoid, input_shape, input_dtype, kernel_name=kernel_name, attrs=attrs)
        accOutput = utils.mod_launch(mod, (input, output), expect=expect)
        return input, accOutput, expect, compare_tensor(accOutput, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape):
    input = random_gaussian(shape, miu=0, sigma=0.5).astype(dtype)
    expect = 1. / (1. + np.exp(-input))
    output = np.full(shape, np.nan, dtype)
    return expect, input, output
