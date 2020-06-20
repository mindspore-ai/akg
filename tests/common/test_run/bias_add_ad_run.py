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
from base import get_rtol_atol
from akg.utils import kernel_exec as utils
from akg.ops.nn import bias_add_ad
from akg.ms.utils import DEFAULT
from akg.utils.result_analysis import np_bisect_sum
from test_op import bias_add_ad_v2
from gen_random import random_gaussian

def gen_data(data_format, dtype, shape):
    """Generate data for testing the op"""
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    head_np = input
    if data_format == "NC1HWC0":
        channel_dims = [1, 4]
    elif data_format == DEFAULT:
        channel_dims = [1]
    else:
        channel_dims = [len(shape) - 1]
    reduce_axis = [i for i in range(len(shape)) if i not in channel_dims]
    if dtype == "float16":
        expect = np_bisect_sum(input, axis=tuple(reduce_axis), keepdims=True)
    else:
        expect = np.sum(input, axis=tuple(reduce_axis), keepdims=True)

    output = np.full(expect.shape, np.nan, dtype)
    return expect, head_np, input, output


def bias_add_ad_run(shape, data_format, dtype,  default_ver=True, attrs=None):
    if attrs == None:
        attrs = {}
    if (len(shape) < 2):
        raise RuntimeError("shape must be 2-d or more dimensions")

    check_list = ["NHWC", "NC1HWC0", DEFAULT]
    if not (data_format in check_list):
        raise RuntimeError("bias_add_grad only support %s while dataformat is %s" % (",".join(check_list), data_format))

    if (data_format == "NHWC"):
        bias = list()
        for i in range(shape[-1]):
            bias.append(float(i))
        bias = np.array(bias).astype(dtype)
        real_shape = shape
    elif (data_format == "DefaultFormat"):
        bias = list()
        for i in range(shape[1]):
            bias.append(float(i))
        bias = np.array(bias).astype(dtype)
        real_shape = shape

    else:
        bias = random_gaussian([1, shape[1], 1, 1, shape[4]], miu=1, sigma=3).astype(dtype)
        real_shape = bias.shape

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        if default_ver:
            mod = utils.op_build_test(bias_add_ad.bias_add_ad, [shape], [dtype], op_attrs=[real_shape, data_format],
                                      kernel_name=kernel_name, attrs=attrs, tuning=t)
        else:
            mod = utils.op_build_test(bias_add_ad_v2.bias_add_ad_v2, [shape], [dtype], op_attrs=[real_shape, data_format],
                                      kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, head_np, input, output = gen_data(data_format, dtype, shape)
            return mod, expect, (head_np, output)
        else:
            return mod
    else:
        if default_ver:
            mod = utils.op_build_test(bias_add_ad.bias_add_ad, [shape], [dtype], op_attrs=[real_shape, data_format],
                                      kernel_name='bias_add_ad', attrs=attrs)
        else:
            mod = utils.op_build_test(bias_add_ad_v2.bias_add_ad_v2, [shape], [dtype], op_attrs=[real_shape, data_format],
                                      kernel_name='bias_add_ad', attrs=attrs)
        expect, head_np, input, output = gen_data(data_format, dtype, shape)
        output = utils.mod_launch(mod, (head_np, output), expect=expect)
        rtol, atol = get_rtol_atol("bias_add_ad", dtype)
        result = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)
        return input, output, expect, result
