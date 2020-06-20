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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import scale
from gen_random import random_gaussian

def scale_run(input_shape, scale_shape, bias_shape, dtype, kernel_name="scale", attrs={}):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        if len(bias_shape) > 0:
            mod = utils.op_build_test(scale.scale_bias, [input_shape, scale_shape, bias_shape], [dtype, dtype, dtype],
                                      kernel_name=kernel_name, attrs=attrs, tuning=t)
        else:
            mod = utils.op_build_test(scale.scale, [input_shape, scale_shape], [dtype, dtype], kernel_name=kernel_name,
                                      attrs=attrs, tuning=t)
        if t:
            bias_data, expect, input_data, output, scale_data = gen_data(bias_shape, dtype, input_shape, scale_shape)
            if len(bias_shape) > 0:
                return mod, expect, (input_data, scale_data, bias_data, output)
            return mod, expect, (input_data, scale_data, output)
        else:
            return mod
    else:
        bias_data, expect, input_data, output, scale_data = gen_data(bias_shape, dtype, input_shape, scale_shape)
        if len(bias_shape) > 0:
            mod = utils.op_build_test(scale.scale_bias, [input_shape, scale_shape, bias_shape], [dtype, dtype, dtype], kernel_name=kernel_name, attrs=attrs)
            output = utils.mod_launch(mod, (input_data, scale_data, bias_data, output), expect=expect)
        else:
            mod = utils.op_build_test(scale.scale, [input_shape, scale_shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs)
            output = utils.mod_launch(mod, (input_data, scale_data, output), expect=expect)

        if len(bias_shape) > 0:
            return (input_data, scale_data, bias_data), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)
        else:
            return (input_data, scale_data), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(bias_shape, dtype, input_shape, scale_shape):
    bias_data = None
    if dtype.lower() in ["float16", "float32", "int32"]:
        input_data = random_gaussian(input_shape, miu=1, sigma=50.0).astype(dtype.lower())
        scale_data = random_gaussian(scale_shape, miu=1, sigma=2.0).astype(dtype.lower())
        if len(bias_shape) > 0:
            bias_data = random_gaussian(bias_shape, miu=1, sigma=0.5).astype(dtype.lower())
    elif dtype.lower() == "int8":
        input_data = np.random.randint(-40, 40, size=input_shape, dtype="int8")
        scale_data = np.random.randint(-3, 3, size=scale_shape, dtype="int8")
        if len(bias_shape) > 0:
            bias_data = np.random.randint(-3, 3, size=bias_shape, dtype="int8")
    elif dtype.lower() == "uint8":
        input_data = np.random.randint(0, 40, size=input_shape, dtype="uint8")
        scale_data = np.random.randint(0, 5, size=scale_shape, dtype="uint8")
        if len(bias_shape) > 0:
            bias_data = np.random.randint(0, 10, size=bias_shape, dtype="uint8")
    else:
        raise RuntimeError("not supported data type %s" % dtype)
    expect = input_data * scale_data
    if len(bias_shape) > 0:
        expect = expect + bias_data
    output = np.full(expect.shape, np.nan, dtype)
    return bias_data, expect, input_data, output, scale_data
