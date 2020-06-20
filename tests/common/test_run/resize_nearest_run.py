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
from test_op import resize_nearest
from .upsampling_run import upsampling_expect
from tensorio import compare_tensor
from gen_random import random_gaussian

def downsampling_expect(input_data, output_shape):
    """
    Integer scale-down. Get the top-left point value of each region.
    """
    scale = [int(input_data.shape[i] / output_shape[i]) for i in range(1, 3)]
    output_data = np.full(output_shape, np.nan, input_data.dtype)
    for i in range(output_shape[1]):
        for j in range(output_shape[2]):
            output_data[:, i, j, :] = input_data[:, i * scale[0], j * scale[1], :]
    return output_data


def non_integer_expect(input, out_shape):
    in_shape = input.shape
    output = np.zeros(out_shape).astype(input.dtype)
    scale_h = 1.0 * in_shape[1] / out_shape[1]
    scale_w = 1.0 * in_shape[2] / out_shape[2]
    for i in range(out_shape[1]):
        for j in range(out_shape[2]):
            old_h = np.floor(i * scale_h).astype("int32")
            old_w = np.floor(j * scale_h).astype("int32")
            output[:, i, j, :] = input[:, old_h, old_w, :]
    return output


def resize_nearest_run(in_shape, out_shape, dtype, kernel_name, attrs):
    kernel_name = utils.gen_name_kernel(kernel_name, dtype, in_shape)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(resize_nearest.resize_nearest,
                                  input_shapes=[in_shape], input_types=[dtype],
                                  op_attrs=[out_shape], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, in_shape, out_shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        # Create op
        mod = utils.op_build_test(resize_nearest.resize_nearest,
                                  input_shapes=[in_shape], input_types=[dtype],
                                  op_attrs=[out_shape], kernel_name=kernel_name, attrs=attrs)
        expect, input, output = gen_data(dtype, in_shape, out_shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)

        return input, output, expect, compare_tensor(output, expect, atol=5e-01, rtol=5e-03, equal_nan=True)


def gen_data(dtype, in_shape, out_shape):
    # Generate data for testing the op
    input = random_gaussian(in_shape, miu=1, sigma=4).astype(dtype)
    # Generate expected output using numpy implementation of resize bilinear
    if np.all([((in_shape[i] >= out_shape[i]) and (in_shape[i] % out_shape[i] == 0)) for i in range(1, 3)]):
        expect = downsampling_expect(input, out_shape)
    elif np.all([((out_shape[i] >= in_shape[i]) and (out_shape[i] % in_shape[i] == 0)) for i in range(1, 3)]):
        expect = upsampling_expect(input, out_shape)
    else:
        expect = non_integer_expect(input, out_shape)
    # Predict output
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input, output
