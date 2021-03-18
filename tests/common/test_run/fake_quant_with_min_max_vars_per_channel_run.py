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

"""fake_quant_with_min_max_vars_per_channel_run"""
import numpy as np

from akg.utils import kernel_exec as utils
from tests.common.test_op import fake_quant_with_min_max_vars_per_channel
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol

def fake_quant_with_min_max_vars_per_channel_run(shape_input, shape_min, shape_max, dtype, num_bits=8,
                                                 narror_range=False, attrs=None):
    """fake_quant_with_min_max_vars_per_channel_run"""
    mod = utils.op_build_test(fake_quant_with_min_max_vars_per_channel.fake_quant_with_min_max_vars_per_channel,
                              [shape_input, shape_min, shape_max], [dtype, dtype, dtype], [num_bits, narror_range],
                              kernel_name='fake_quant_with_min_max_vars_per_channel', attrs=attrs)
    args, exp_output, input_data, input_min, input_max = gen_data(shape_input, shape_min, shape_max, dtype, num_bits,
                                                                  narror_range)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("fake_quant_with_min_max_vars_per_channel", dtype)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)
    return [input_data, input_min, input_max], acu_output, exp_output, testcase_result

def _nudged_min_max_compute(input_min, input_max, num_bits, narrow_range):
    # Now only supports quantization when min <= 0 <= max.
    quant_min = 1 if narrow_range else 0
    quant_max = (2 ** num_bits) - 1
    scale = (input_max - input_min) / (quant_max - quant_min)
    nudged_min = scale * np.floor(input_min / scale + 0.5)
    nudged_max = input_max + nudged_min - input_min

    return nudged_min, nudged_max, scale

def gen_data(shape_input, shape_min, shape_max, dtype, num_bits, narror_range):
    """generate valid data to test"""
    input_data = random_gaussian(shape_input, miu=10, sigma=0.3).astype(dtype)
    input_min = random_gaussian(shape_min, miu=-6, sigma=0.3).astype(dtype)
    input_max = random_gaussian(shape_max, miu=6, sigma=0.3).astype(dtype)

    nudged_min_nudged_max = _nudged_min_max_compute(input_min, input_max, num_bits, narror_range)

    clamped = np.maximum(np.minimum(input_data, nudged_min_nudged_max[1]), nudged_min_nudged_max[0])
    clamped_shifted_div_scale = (clamped - nudged_min_nudged_max[0]) / nudged_min_nudged_max[2]

    result_round = np.floor(clamped_shifted_div_scale + 0.5)
    tmp_res = result_round * nudged_min_nudged_max[2] + nudged_min_nudged_max[0]

    exp_output = tmp_res * np.not_equal((np.abs(input_min) + np.abs(input_max)), 0.0)
    # inputs and output to hold the data
    output = np.full(exp_output.shape, np.nan, dtype)
    args = [input_data, input_min, input_max, output]
    return args, exp_output, input_data, input_min, input_max
