# Copyright 2020 Huawei Technologies Co., Ltd
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
from test_op import fake_quant_with_min_max_args_gradient
from test_op.fake_quant_with_min_max_args import nudge_min_max
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian

def fake_quant_with_min_max_args_gradient_execute(shape, dtype, min, max, num_bits, narrow_range, attrs):
    exp_output, inputs, args = gen_data(dtype, shape, min, max, num_bits, narrow_range)
    mod = fake_quant_with_min_max_args_gradient_compile(shape, dtype, min, max, num_bits, narrow_range, attrs)
    # result_tvm
    acu_output = utils.mod_launch(mod, args, expect=exp_output)

    # compare result
    rtol, atol = get_rtol_atol("fake_quant_with_min_max_args_gradient", dtype)
    TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shape, min, max, num_bits, narrow_range):
    # Result_Numpy
    head = random_gaussian(shape, miu=0, sigma=0.3).astype(dtype)
    inputs = random_gaussian(shape, miu=0, sigma=0.3).astype(dtype)
    nudged_min, nudged_max, _ = nudge_min_max(min, max, num_bits, narrow_range)
    #where((inputs<=nudged_max)&(x>=nudged_min),1,0)
    leq = np.less_equal(inputs, nudged_max)
    geq = np.less_equal(nudged_min, inputs)
    inrange = np.multiply(leq, geq)    
    exp_output = np.multiply(head, inrange)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = []
    args.append(head)
    args.append(inputs)
    args.append(output)
    return exp_output, inputs, args


def fake_quant_with_min_max_args_gradient_compile(shape, dtype, min, max, num_bits, narrow_range, attrs,
                                                  kernel_name='fake_quant_with_min_max_args_gradient', runing=False):
    return utils.op_build_test(fake_quant_with_min_max_args_gradient.fake_quant_with_min_max_args_gradient, [shape, shape],
                               [dtype, dtype], [min, max, num_bits, narrow_range], kernel_name, attrs=attrs, tuning=runing)

