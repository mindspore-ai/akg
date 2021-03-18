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

""" test_fused_relu_grad """
from __future__ import absolute_import
import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian
from tests.common.test_op.resnet.fused_relu_grad import fused_relu_grad


def gen_data(shape, dtype):
    data1 = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    data2 = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    data3 = random_gaussian(shape, miu=0, sigma=1).astype(dtype)

    return [data1, data2, data3]


def compute_expect(input, c1):
    shape = input[0].shape
    dtype = input[0].dtype
    data_zero = np.full(shape, c1, dtype)
    cmp_zero = np.greater(input[2], data_zero)
    data_add = np.add(input[0], input[1])

    return np.where(cmp_zero, data_add, data_zero)


def test_fused_relu_grad(shape, c1=0, poly_sch=False):
    dtype = 'float16'
    input = gen_data(shape, dtype)
    expect = compute_expect(input, c1)
    shapes = [shape] * 3
    dtypes = [dtype] * 3
    attrs = [c1]
    if poly_sch:
        mod = utils.op_build_test(fused_relu_grad, shapes, dtypes, op_attrs=attrs, kernel_name="fused_relu_grad",
                                  attrs={"target": "cuda"})

    output = np.full(shape, np.nan, dtype)
    output = utils.mod_launch(mod, (*input, output), expect=expect)
    res = np.allclose(output, expect, rtol=5e-3, atol=1e-8)
    print("Test {}".format("Pass" if res else "Failed"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    return True
