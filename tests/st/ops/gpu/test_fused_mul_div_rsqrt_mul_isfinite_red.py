# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
# limitations under the License

from __future__ import absolute_import
import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from tests.common.test_op.resnet.fused_mul_div_rsqrt_mul_isfinite_red import fused_mul_div_rsqrt_mul_isfinite_red


def gen_data(shape, dtype):
    data1 = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    data2 = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    return [data1, data2]


def compute_expect(input):
    mul_param1 = np.multiply(input[1], input[1])
    divide_val = np.divide(1, mul_param1)

    sqrt_val = np.sqrt(divide_val)
    rsqrt_val = np.divide(1, sqrt_val)
    mul_param0 = np.multiply(input[0], rsqrt_val)

    isfinite = np.isfinite(mul_param0)
    reduce_logic_and = np.logical_and.reduce(isfinite)
    reduce_logic_and = np.full((1,), reduce_logic_and, 'bool')

    return [reduce_logic_and, mul_param0, rsqrt_val, divide_val]


def test_fused_mul_div_rsqrt_mul_isfinite_red(shape, dtype='float32', poly_sch=False):
    input = gen_data(shape, dtype)
    expect = compute_expect(input)
    input_shape = [shape, shape]
    input_dtype = [dtype, dtype]
    if poly_sch:
        mod = utils.op_build_test(fused_mul_div_rsqrt_mul_isfinite_red, input_shape, input_dtype,
                                  kernel_name="fused_mul_div_rsqrt_mul_isfinite_red",
                                  attrs={"target": "cuda", "enable_akg_reduce_lib": True, "enable_atomic_add": True})

    outputs = [np.full((1,), False, 'bool')] + [np.full(shape, np.nan, dtype)] * 3
    output = utils.mod_launch(mod, [*input, *outputs], outputs=list(range(-len(outputs), 0)), expect=expect)
    ret = compare_tensor(output[0], expect[0], rtol=5e-03, atol=1.e-08)
    ret &= compare_tensor(output[1], expect[1], rtol=5e-03, atol=1.e-08)
    ret &= compare_tensor(output[2], expect[2], rtol=5e-03, atol=1.e-08)
    ret &= compare_tensor(output[3], expect[3], rtol=5e-03, atol=1.e-08)
    print("Test {}".format("Pass" if ret else "Failed"))
    if not ret:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    return True
