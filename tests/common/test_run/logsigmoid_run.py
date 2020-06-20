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
from test_op import logsigmoid
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian

def logsigmoid_run(shape, dtype, kernel_name, attrs):
    input_shape = [shape]
    input_dtype = [dtype]
    
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(logsigmoid.logsigmoid, input_shape, input_dtype,
                                  kernel_name=kernel_name,
                                  attrs=attrs,
                                  tuning=t)
        if t:
            expect, input, output = gen_data(dtype, shape)
            return mod, expect, (input, output)
        else:
            return mod

    else:
        expect, input, output = gen_data(dtype, shape)
        mod = utils.op_build_test(logsigmoid.logsigmoid, input_shape, input_dtype, kernel_name=kernel_name, attrs=attrs)
        output = utils.mod_launch(mod, (input, output), expect=expect)
        rtol, atol = get_rtol_atol("logsigmoid", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape):
    input_np = random_gaussian(shape, miu=0, sigma=0.5).astype(dtype)
    expect = np.log(1. / (1. + np.exp(-input_np)))
    output = np.full(shape, np.nan, dtype)
    return expect, input_np, output