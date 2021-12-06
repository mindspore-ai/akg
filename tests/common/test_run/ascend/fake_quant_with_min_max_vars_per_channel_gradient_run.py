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

"""fake_quant_with_min_max_vars_per_channel_gradient_run"""
import numpy as np

from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import fake_quant_with_min_max_vars_per_channel_gradient
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol

def fake_quant_with_min_max_vars_per_channel_gradient_run(shape_gradient, shape_input, shape_min, shape_max, dtype, 
                                                          num_bits=8, narror_range=False, attrs=None):
    """fake_quant_with_min_max_vars_per_channel_gradient_run"""
    mod = utils.op_build_test(fake_quant_with_min_max_vars_per_channel_gradient.fake_quant_with_min_max_vars_per_channel_gradient,
                              [shape_gradient, shape_input, shape_min, shape_max], [dtype, dtype, dtype, dtype],
                              [num_bits, narror_range], kernel_name='fake_quant_with_min_max_vars_per_channel_gradient', attrs=attrs)
    args, exp_output, input_gradient, input_data, input_min, input_max = gen_data(shape_gradient, shape_input, shape_min, shape_max,
                                                                                  dtype, num_bits, narror_range)
    acu_output = utils.mod_launch(mod, args, expect=exp_output, outputs=(-3, -2, -1))
    # compare result
    rtol, atol = get_rtol_atol("fake_quant_with_min_max_vars_per_channel_gradient", dtype)
    testcase_result_0 = compare_tensor(acu_output[0], exp_output[0], rtol=rtol, atol=atol, equal_nan=True)
    testcase_result_1 = compare_tensor(acu_output[1], exp_output[1], rtol=rtol, atol=atol, equal_nan=True)
    testcase_result_2 = compare_tensor(acu_output[2], exp_output[2], rtol=rtol, atol=atol, equal_nan=True)
    testcase_result = list(map(lambda x,y: compare_tensor(x, y, rtol=rtol, atol=atol), acu_output, exp_output))
    return [input_gradient, input_data, input_min, input_max], acu_output, exp_output, all(testcase_result)

def _nudged_min_max_compute(input_min, input_max, num_bits, narrow_range):
    # Now only supports quantization when min <= 0 <= max.
    quant_min = 1 if narrow_range else 0
    quant_max = (2 ** num_bits) - 1
    scale = (input_max - input_min) / (quant_max - quant_min)
    nudged_min = scale * np.floor(input_min / scale + 0.5)
    nudged_max = input_max + nudged_min - input_min

    return nudged_min, nudged_max, scale

def gen_data(shape_gradient, shape_input, shape_min, shape_max, dtype, num_bits, narror_range):
    """generate valid data to test"""
    input_gradient = random_gaussian(shape_input, miu=10, sigma=0.3).astype(dtype)
    input_data = random_gaussian(shape_input, miu=0, sigma=3).astype(dtype)
    input_min = random_gaussian(shape_min, miu=-6, sigma=0.3).astype(dtype)
    input_max = random_gaussian(shape_max, miu=6, sigma=0.3).astype(dtype)

    nudged_min, nudged_max, _ = _nudged_min_max_compute(input_min, input_max, num_bits, narror_range)
    # both zero yeilds zero
    bool_both_zero_value = np.subtract(1, np.logical_and(np.equal(input_min, 0), np.equal(input_max, 0)))
    bool_both_zero_value = np.broadcast_to(bool_both_zero_value, shape_input)
    bool_both_zero_negate = np.subtract(1, bool_both_zero_value)
    bool_between_nudged_min_max = np.subtract(1, np.logical_or(np.less(input_data, nudged_min), np.greater(input_data, nudged_max)))
 
    sum_axis = tuple(x for x in range(0, len(shape_input)-1))
    # gradient is 1 if input in [min, max] else 0
    backprops_input_tmp = np.multiply(bool_between_nudged_min_max, input_gradient)
    backprops_bool_both_zero = np.multiply(backprops_input_tmp, bool_both_zero_value)
    # if min and max are both zero, gradients is input_gradients
    input_gradients_both_zero = np.multiply(input_gradient, bool_both_zero_negate)
    backprops_input = np.add(backprops_bool_both_zero, input_gradients_both_zero)

    # gradients for min is input_gradients if inputs_data < nudged_min else 0
    bool_less_nudged_min = np.less(input_data, nudged_min)
    output_backprop_min_tmp = np.multiply(bool_less_nudged_min, input_gradient)
    # gradients for min is 0 if min and max are both 0
    output_backprop_min_bool = np.multiply(output_backprop_min_tmp, bool_both_zero_value)
    if sum_axis == []:
        output_backprop_min = output_backprop_min_bool
    else:
        output_backprop_min = np.sum(output_backprop_min_bool, sum_axis)

    # gradients for max is input_gradients if inputs_data > nudged_max else 0
    bool_more_nudged_max = np.greater(input_data, nudged_max)
    output_backprop_max_tmp = np.multiply(bool_more_nudged_max, input_gradient)
    # gradients for max is 0 if min and max are both 0
    output_backprop_max_bool = np.multiply(output_backprop_max_tmp, bool_both_zero_value)
    if sum_axis == []:
        output_backprop_max = output_backprop_max_bool
    else:
        output_backprop_max = np.sum(output_backprop_max_bool, sum_axis)

    exp_output = [backprops_input, output_backprop_min, output_backprop_max]
    # inputs and output to hold the data
    output_input = np.full(shape_input, np.nan, dtype)
    output_min = np.full(shape_min, np.nan, dtype)
    output_max = np.full(shape_max, np.nan, dtype)
    args = [input_gradient, input_data, input_min, input_max, output_input, output_min, output_max]
    return args, exp_output, input_gradient, input_data, input_min, input_max
