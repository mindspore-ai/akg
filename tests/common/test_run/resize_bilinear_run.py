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
from tests.common.test_op import resize_bilinear
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def resize_bilinear_run(in_shape, out_shape, dtype, kernel_name, attrs):
    kernel_name = utils.gen_name_kernel(kernel_name, dtype, in_shape)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(resize_bilinear.resize_bilinear,
                                  input_shapes=[in_shape], input_types=[dtype],
                                  op_attrs=[out_shape], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, in_shape, out_shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        # Create op
        mod = utils.op_build_test(resize_bilinear.resize_bilinear,
                                  input_shapes=[in_shape], input_types=[dtype],
                                  op_attrs=[out_shape], kernel_name=kernel_name, attrs=attrs)

        expect, input, output = gen_data(dtype, in_shape, out_shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)
        return input, output, expect, compare_tensor(output, expect, atol=5e-01, rtol=5e-03, equal_nan=True)


def gen_data(dtype, in_shape, out_shape):
    # Generate data for testing the op
    input = random_gaussian(in_shape, miu=1, sigma=4).astype(dtype)
    # Generate expected output using numpy implementation of resize bilinear
    expect = bilinear_expect(input, out_shape)
    # Predict output
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input, output


def bilinear_expect(input_data, out_shape):
    # Get N,W,H,C from input data
    batch_size, in_height, in_width, channels = input_data.shape
    out_height, out_width = out_shape[0], out_shape[1]
    out_shape = [batch_size, out_height, out_width, channels]

    # scale value is required to map from input space to output space
    # align_corner version:
    height_scale = (in_height - 1.0) / (out_height - 1.0)
    width_scale = (in_width - 1.0) / (out_width - 1.0)

    # compute_interpolation_weights calculates lower, upper and lerp for each index of ys and xs
    def compute_interpolation_weights(index, scale):
        temp = CachedInterpolation(0, 0, 0)
        temp.lower = np.floor(index * scale).astype("int32")
        temp.upper = np.ceil(index * scale).astype("int32")
        temp.lerp = index * scale - temp.lower
        return temp

    # ys and xs will ensure which row(top,bottom) and column(left,right) index from input matrix will be responsible for the wighted calculation for each position of output matrix
    # ys will provide row information and xs will provide column information for output matrix
    # so ys size will be same as output height and xs size will be same as output width
    # they will also contain interpolation weight(lerp)
    ys = [compute_interpolation_weights(i, height_scale) for i in range(out_height)]
    xs = [compute_interpolation_weights(i, width_scale) for i in range(out_width)]

    return resize_image(input_data, out_shape, xs, ys)

# each position of row and column index of output matrix will contain lower, upper and lerp


class CachedInterpolation:
    def __init__(self, lower, upper, lerp):
        self.lower = lower
        self.upper = upper
        self.lerp = lerp


def resize_image(input_data, out_shape, xs, ys):
    def compute_lerp(top_left, top_right, bottom_left, bottom_right, x_lerp, y_lerp):
        top = top_left + (top_right - top_left) * x_lerp
        bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
        return top + (bottom - top) * y_lerp

    output = np.zeros(out_shape).astype(input_data.dtype)
    batch_size, out_height, out_width, channels = out_shape
    for b in range(batch_size):
        for y in range(out_height):
            for x in range(out_width):
                for c in range(channels):
                    left_x_index = xs[x].lower
                    right_x_index = xs[x].upper
                    xs_lerp = xs[x].lerp

                    top_y_index = ys[y].lower
                    bottom_y_index = ys[y].upper
                    ys_lerp = ys[y].lerp

                    top_left = input_data[b][top_y_index][left_x_index][c]
                    top_right = input_data[b][top_y_index][right_x_index][c]
                    bottom_left = input_data[b][bottom_y_index][left_x_index][c]
                    bottom_right = input_data[b][bottom_y_index][right_x_index][c]

                    output[b][y][x][c] = compute_lerp(top_left, top_right, bottom_left, bottom_right, xs_lerp, ys_lerp)
    return output
