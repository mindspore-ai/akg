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
from tests.common.test_op import upsampling
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
'''
@param input_data: original image with data_format "NHWC".
@param output_shape: output image shape with data_format "NHWC".
    notice that BatchNum and ChannelNum of output shape must be equal to input shape.
'''


def upsampling_expect(input_data, output_shape):
    scale = [output_shape[i] / input_data.shape[i] for i in range(1, 3)]
    tmp = np.repeat(input_data, scale[0], axis=1)
    return np.repeat(tmp, scale[1], axis=2)


def upsampling_run(in_shape, out_shape, dtype, kernel_name, attrs):
    kernel_name = utils.gen_name_kernel(kernel_name, dtype, in_shape)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(upsampling.upsampling,
                                  input_shapes=[in_shape], input_types=[dtype],
                                  op_attrs=[out_shape], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, in_shape, out_shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        # Create op
        mod = utils.op_build_test(upsampling.upsampling,
                                  input_shapes=[in_shape], input_types=[dtype],
                                  op_attrs=[out_shape], kernel_name=kernel_name, attrs=attrs)
        expect, input, output = gen_data(dtype, in_shape, out_shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)

        return input, output, expect, compare_tensor(output, expect, atol=5e-01, rtol=5e-03, equal_nan=True)


def gen_data(dtype, in_shape, out_shape):
    # Generate data for testing the op
    input = random_gaussian(in_shape, miu=1, sigma=4).astype(dtype)
    # Generate expected output using numpy implementation of resize bilinear
    expect = upsampling_expect(input, out_shape)
    # Predict output
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input, output
