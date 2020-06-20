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

"""sigmoid_cross_entropy_with_logits_grad_run"""
import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import sigmoid_cross_entropy_with_logits_grad
from base import get_rtol_atol
from gen_random import random_gaussian

def sigmoid_cross_entropy_with_logits_grad_run(shape, dtype, attrs):
    """sigmoid_cross_entropy_with_logits_grad_run implementation"""
    mod = utils.op_build_test(sigmoid_cross_entropy_with_logits_grad.sigmoid_cross_entropy_with_logits_grad,
                              [shape, shape, shape], [dtype, dtype, dtype],
                              kernel_name='sigmoid_cross_entropy_with_logits_grad', op_attrs=[], attrs=attrs)
    args, exp_output, predict, target, dout = gen_data(dtype, shape)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("sigmoid_cross_entropy_with_logits_grad", dtype)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return [predict, target, dout], acu_output, exp_output, testcase_result



def gen_data(dtype, shape):
    # generate data for test
    predict = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    target = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    dout = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    exp_output = ((1 / (np.exp(-predict) + 1)) - target) * dout
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = [predict, target, dout, output]
    return args, exp_output, predict, target, dout
