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

"""bias_add_run"""

import math
import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.nn import bias_add
from tensorio import compare_tensor
from gen_random import random_gaussian
from base import get_rtol_atol
from test_utils import compute_blockdim


def bias_add_run(shape, data_format, dtype, attrs):
    """run function for dsl function bias_add."""
    if attrs is None:
        attrs = {}
    if data_format == "NHWC":
        bias_shape = (shape[-1], )
    elif data_format == "DefaultFormat":
        if len(shape) == 2:
            bias_shape = (shape[-1], )
        elif len(shape) == 4:
            # NCHW
            bias_shape = (shape[1], )
        else:
            raise RuntimeError("bias_add only support 2D and 4D shape while dataformat is DefaultFormat")
    else:
        # NC1HWC0
        bias_shape = [1, shape[1], 1, 1, shape[4]]
    bias = random_gaussian(bias_shape).astype(dtype)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(bias_add.bias_add,
                                  [shape, bias_shape], [dtype, dtype],
                                  kernel_name=kernel_name, op_attrs=[data_format], attrs=attrs, tuning=t)
        if t:
            expect, inputs, output = gen_data(bias, dtype, shape, data_format)
            return mod, expect, (inputs, bias, output)

        return mod
    if 'mod' in attrs.keys():
        mod = attrs["mod"]
    else:
        mod = utils.op_build_test(bias_add.bias_add, [shape, bias_shape], [dtype, dtype], kernel_name='bias_add',
                                  op_attrs=[data_format], attrs=attrs)
    expect, inputs, output = gen_data(bias, dtype, shape, data_format)
    args = [inputs, bias, output]
    if attrs.get("dynamic"):
        for i in shape:
            args.append(i)
        block_dim = compute_blockdim(shape)
        args.append(block_dim)
    output = utils.mod_launch(mod, args, outputs=(2,), expect=expect)
    rtol, atol = get_rtol_atol("bias_add", dtype)
    return (inputs, bias), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(bias, dtype, shape, data_format):
    """Generates input, output and expect data."""
    # Generate data for testing the op
    inputs = random_gaussian(shape, miu=1, sigma=3).astype(dtype)
    if data_format == "NHWC":
        bias = np.reshape(bias, (1, 1, 1, bias.shape[0]))
    elif data_format == "DefaultFormat":
        if len(shape) == 2:
            bias = np.reshape(bias, (1, bias.shape[0]))
        else:
            bias = np.reshape(bias, (1, bias.shape[0], 1, 1))
    expect = inputs + bias
    output = np.full(expect.shape, np.nan, dtype)
    return expect, inputs, output
