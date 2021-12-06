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

"""run function for fake_quant_with_min_max_args"""

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend.fake_quant_with_min_max_args import fake_quant_with_min_max_args
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol

def fake_quant_with_min_max_args_run(shape, dtype, min, max, num_bits, 
                                     narrow_range, attrs):
    mod = utils.op_build_test(fake_quant_with_min_max_args, [shape],
                              [dtype], [min, max, num_bits, narrow_range],
                              kernel_name="fake_quant_with_min_max_args", 
                              attrs=attrs)
    expect, inputs, output = gen_data(shape, dtype, min, max, num_bits, 
                                      narrow_range)
    output = utils.mod_launch(mod, (inputs, output), expect=expect)
    rtol, atol = get_rtol_atol("fake_quant_with_min_max_args", dtype)
    TestCase_Result = compare_tensor(
        output, expect, rtol=rtol, atol=atol, equal_nan=False)
    return inputs, output, expect, TestCase_Result

def gen_data(shape, dtype, min, max, num_bits, narrow_range):
    input_data = np.random.uniform(-2, 2, size=shape).astype(dtype)
    nudged_min, nudged_max, scale = _nudged_min_max(min, max, num_bits,
                                                    narrow_range)
    zero_tensor = np.broadcast_to(0.0, shape)
    nudged_max_tensor = zero_tensor + nudged_max
    nudged_min_tensor = zero_tensor + nudged_min
    inv_nudged_scale = 1.00 / scale

    # Transform the input between nudged_max and nudged_min
    clamped_vmin = np.minimum(input_data, nudged_max_tensor)
    clamped = np.maximum(clamped_vmin, nudged_min_tensor)

    # Calculate the quantized and dequantized results
    clamped_shifted = clamped - nudged_min_tensor
    vmul_shifted = clamped_shifted * inv_nudged_scale
    vadds_shifted = vmul_shifted + 0.5
    floor_cast = np.floor(vadds_shifted)
    res_scale = floor_cast * scale
    res = res_scale + nudged_min_tensor
    expect = res
    output = np.full(shape, np.nan, dtype)
    return expect, input_data, output

def _nudged_min_max(min, max, num_bits, narrow_range):
    quant_max = (2**num_bits) - 1

    if narrow_range is False:
        quant_min = 0.00
    else:
        quant_min = 1.00

    scale = (max - min) / (float(quant_max) - quant_min)

    zeor_point_from_min = quant_min - min / scale

    # Calculate the maximum and minimum values of the quantization
    if zeor_point_from_min < quant_min:
        nudged_zero_point = quant_min
    elif zeor_point_from_min > quant_max:
        nudged_zero_point = quant_max
    else:
        nudged_zero_point = (zeor_point_from_min + 0.5) // 1

    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale

    return nudged_min, nudged_max, scale

