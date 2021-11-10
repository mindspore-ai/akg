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

"""operator dsl function: fake_quant_with_min_max_args"""

import akg
from akg import tvm, topi
from akg.utils.format_transform import get_shape
import akg.utils as utils
from akg.ops.math.ascend import Floor

def nudge_min_max(min, max, num_bits, narrow_range):
    """
    Calculate the maximum and minimum values of the quantization

    Args:
        min: scalar, input min
        max: input max
        num_bits: scalar
        Defaults to 8. num_bits is the bitwidth of the quantization, range [2,16]
        narrow_range: bool

    Returns:
        nudged_min, nudged_max, scale
    """
    quant_max = (2**num_bits) - 1

    if narrow_range is False:
        quant_min = 0.00
    else:
        quant_min = 1.00

    scale = (max - min) / (float(quant_max) - quant_min)

    zero_point_from_min = quant_min - min / scale

    # Calculate the maximum and minimum values of the quantization
    if zero_point_from_min < quant_min:
        nudged_zero_point = quant_min
    elif zero_point_from_min > quant_max:
        nudged_zero_point = quant_max
    else:
        nudged_zero_point = (zero_point_from_min + 0.5) // 1

    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale

    return nudged_min, nudged_max, scale


@utils.check_input_type(tvm.tensor.Tensor,
                          (float, int, type(None)),
                          (float, int, type(None)),
                          (int, type(None)), 
                          (bool, type(None)))
def fake_quant_with_min_max_args(input_data, min=-6, max=6, num_bits=8, 
                                 narrow_range=False):
    """
    Computes Fake-quantize the 'input_data' tensor,
    type float32 to 'output_data' tensor of same type

    output_data = (floor(clamped_shifted * inv_nudged_scale + 0.5f))) * scale
                  + nudged_min
    scale = (max-min) / (quant_max-quant_min)

    Args:
        data_x1 (tvm.tensor.Tensor): Tensor of dtype "float32"
        min ([float, int]): scalar, defaults to -6
        max ([float, int]): scalar, defaults to 6. [min; max] define the 
                            clamping range for the input_data data
        num_bits ([float, int]): Defaults to 8. num_bits is the bitwidth
                                 of the quantization,between 2 and 16
        narrow_range ([bool]): 
            True, quantized into the quantization range [1; 2^num_bits - 1]
            False,quantized into the quantization range [0; 2^num_bits - 1]

    Returns:
        tvm.tensor.Tensor
    """
    shape = get_shape(input_data)
    utils.check_shape(shape)

    dtype = input_data.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.FLOAT32)

    nudged_min, nudged_max, scale = nudge_min_max(min, max, num_bits,
                                                  narrow_range)

    zero_tensor = tvm.compute(input_data.shape, 
                              lambda *i: tvm.const(0, dtype="float32"),
                              name="zero_tensor")
    nudged_max_tensor = topi.add(zero_tensor, nudged_max)
    nudged_min_tensor = topi.add(zero_tensor, nudged_min)
    inv_nudged_scale = 1.00 / scale

    # Transform the input between nudged_max and nudged_min
    clamped_vmin = topi.minimum(input_data, nudged_max_tensor)
    clamped = topi.maximum(clamped_vmin, nudged_min_tensor)

    # Calculate the quantized and dequantized results
    clamped_shifted = topi.subtract(clamped, nudged_min_tensor)
    vmul_shifted = topi.multiply(clamped_shifted, inv_nudged_scale)
    vadds_shifted = topi.add(vmul_shifted, 0.5)
    floor_vadds_shifted = Floor(vadds_shifted)
    floor_cast = akg.lang.ascend.cast_to(floor_vadds_shifted, dtype)
    res_scale = topi.multiply(floor_cast, scale)
    res = topi.add(res_scale, nudged_min_tensor)

    return res
