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
# limitations under the License.

import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.math.ascend import AccumulateNv2
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def accumulate_nv2_execute(shape, dtype, n, attrs=None):
    if attrs is None:
        attrs = {}
    mod, shapes = accumulate_nv2_compile(shape, dtype, n, attrs)
    exp_output, inputs, args = gen_data(dtype, shapes)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    rtol, atol = get_rtol_atol("accumulate_nv2", dtype)
    TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)
    return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shapes):
    inputs = [random_gaussian(s, miu=1, sigma=0.1).astype(dtype) for s in shapes]
    if len(shapes) == 1:
        exp_output = inputs[0]
    else:
        exp_output = np.sum(inputs, axis=0)
    output = np.full(exp_output.shape, np.nan, dtype)
    args = [*inputs, output]
    return exp_output, inputs, args


def accumulate_nv2_compile(shape, dtype, n, attrs, kernel_name="accumulate_nv2", tuning=False):
    shapes = [shape] * n
    return utils.op_build_test(AccumulateNv2, [shapes], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning), shapes
