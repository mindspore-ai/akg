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

import akg.tvm
import numpy as np
from test_op import resize_bilinear_grad
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
import math
from tensorio import compare_tensor


def resize_bilinear_grad_run(resized_shape, original_shape, dtype, kernel_name, attrs):
    kernel_name = utils.gen_name_kernel(kernel_name, dtype, resized_shape)
    original_data = akg.tvm.placeholder(original_shape, dtype)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(resize_bilinear_grad.resize_bilinear_grad,
                                  [resized_shape], [dtype], op_attrs=[original_data],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, original_shape, resized_shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(resize_bilinear_grad.resize_bilinear_grad,
                                  [resized_shape], [dtype], op_attrs=[original_data],
                                  kernel_name=kernel_name, attrs=attrs)
        expect, input, output = gen_data(dtype, original_shape, resized_shape)
        # auto-tensor output
        output = utils.mod_launch(mod, (input, output), expect=expect)
        return input, output, expect, compare_tensor(output, expect, atol=5e-01, rtol=5e-03, equal_nan=True)


def gen_data(dtype, original_shape, resized_shape):
    # Generate data for testing the op
    input = random_gaussian(resized_shape, miu=10, sigma=0.1).astype(dtype)
    output = np.full(original_shape, np.nan, dtype)
    # Generate expected output using numpy implementation of resize bilinear
    expect = bilinear_grad_expect(input, original_shape)
    return expect, input, output


def bilinear_grad_expect(input_grad, output_shape):
    batch, original_height, original_width, channels = output_shape
    resized_height, resized_width = input_grad.shape[1:3]
    output_grad = np.zeros(output_shape)

    height_scale = (original_height - 1.0) / (resized_height - 1.0)
    width_scale = (original_width - 1.0) / (resized_width - 1.0)

    for b in range(batch):
        for y in range(resized_height):
            in_y = y * height_scale
            top_y_index = int(max(math.floor(in_y), 0))
            bottom_y_index = int(min(math.ceil(in_y), original_height - 1))
            y_lerp = in_y - math.floor(in_y)
            inverse_y_lerp = 1.0 - y_lerp
            for x in range(resized_width):
                in_x = x * width_scale
                left_x_index = int(max(math.floor(in_x), 0))
                right_x_index = int(min(math.ceil(in_x), original_width - 1))
                x_lerp = in_x - math.floor(in_x)
                inverse_x_lerp = 1.0 - x_lerp
                for c in range(channels):
                    output_grad[b, top_y_index, left_x_index, c] += input_grad[b, y, x, c] * inverse_y_lerp * inverse_x_lerp
                    output_grad[b, top_y_index, right_x_index, c] += input_grad[b, y, x, c] * inverse_y_lerp * x_lerp
                    output_grad[b, bottom_y_index, left_x_index, c] += input_grad[b, y, x, c] * y_lerp * inverse_x_lerp
                    output_grad[b, bottom_y_index, right_x_index, c] += input_grad[b, y, x, c] * y_lerp * x_lerp
    return output_grad
