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

"""operator dsl function: fake_quant_with_min_max_vars_per_channel"""
import akg.lang.cce
from akg import tvm, topi
from akg.utils import validation_check as vc_util
from akg.utils import dsl_create as dc
from akg.utils.format_transform import get_shape
from akg.ops.math.div import div
from akg.ops.math.reciprocal import reciprocal
from akg.ops.math.mul import mul
from akg.utils import kernel_exec as utils

def less_compare_float32(data_x, data_y):
    """if x is less than y, then return 1, else return 0"""
    shape_inputs = get_shape(data_x)
    # minimun num of float32 2**(-126)
    data_min = akg.lang.cce.broadcast(tvm.const(2**(-126), dtype="float32"), shape_inputs, "float32")
    data_zero = akg.lang.cce.broadcast(dc.zero_const("float32"), shape_inputs, "float32")
    res_sub = topi.subtract(data_y, data_x)
    res_min = topi.minimum(res_sub, data_min)
    res_max = topi.maximum(res_min, data_zero)
    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 62 * 62 * 2 to adaptor 126
    res_mul_fierst = topi.multiply(res_max, tvm.const(2**62, dtype="float32"))
    res_mul_second = topi.multiply(res_mul_fierst, tvm.const(2**62, dtype="float32"))
    res = topi.multiply(res_mul_second, tvm.const(2**2, dtype="float32"))

    return res

def nudged_min_max_compute(min_broadcast, max_broadcast, num_bits, narrow_range):
    """
    Calculate the maximum and minimum values of the quantization.

    Notes:
        Each channel scale[i] euqal to (max_broadcast[i] - min_broadcast[i]) / (quant_max - quant_min).
        Then compute nudged_zero_point:
                nudged_zero_point = floor(between_min_max_float + 0.5) + less_quant_min_float + more_quant_max_float,
        between_min_max_float is first calculated by:
                zero_point_from_min = (quant_min_float - min_broadcast) / scale,
        then between_min_max_float = zero_point_from_min, which min_broadcast <= zero_point_from_min <= max_broadcast.
        Besides, the value of less_quant_min_float is equal to quant_min or zero, zero_point_from_min < quant_min_float,
        the value is quant_min, else is 0. The same as more_quant_max_float.
        Finally according to scale and nudged_zero_point to compute nudged_min and nudged_max:
                 nudged_min = (quant_min - nudged_zero_point) * scale
                 nudged_max = (quant_max - nudged_zero_point) * scale

    Args:
        min_broadcast (tvm.tensor.Tensor): minimum value to be quantified for each channel.
        max_broadcast (tvm.tensor.Tensor): maximum value to be quantified for each channel.
        num_bits (int): num_bits is the bitwidth of the quantization, range [2,16].
        narrow_range (bool): if True, for each channel, quantized into the quantization range [0, 2^num_bits - 1] else
                      quantized into the quantization range [1, 2^num_bits - 1].

    Returns:
        nudged_min (tvm.tensor.Tensor): The same type and shape as min_broadcast.
        nudged_max (tvm.tensor.Tensor): The same type and shape as max_broadcast.
        scale (tvm.tensor.Tensor): The same type and shape as max_broadcast.
    """


    dtype = min_broadcast.dtype
    quant_min = 1 if narrow_range else 0
    quant_max = (2 ** num_bits) - 1

    # because of need compute each channel, so quant_min and quant_max need to broadcast.
    quant_min_float = topi.full(min_broadcast.shape, dtype, tvm.const(quant_min, dtype))
    quant_max_float = topi.full(min_broadcast.shape, dtype, tvm.const(quant_max, dtype))

    # caculate each channel max and min difference.
    max_sub_min = topi.subtract(max_broadcast, min_broadcast)
    quant_max_sub_quant_min = topi.subtract(quant_max_float, quant_min_float)
    # compute scale = (max_broadcast - min_broadcast) / (quant_max - quant_min)
    # and min_div_scale = min_broadcast / scale
    if utils.product_is_mini():
        scale = mul(max_sub_min, reciprocal(quant_max_sub_quant_min))
        min_div_scale = mul(min_broadcast, reciprocal(scale))
    else:
        scale = div(max_sub_min, quant_max_sub_quant_min)
        min_div_scale = div(min_broadcast, scale)

    # zero_point_from_min = quant_min_float - min_broadcast / scale
    zero_point_from_min = topi.subtract(quant_min_float, min_div_scale)
    # if zero_point_from_min < quant_min_float, bool_less_quant_min_float = 1 else 0
    bool_less_quant_min_float = less_compare_float32(zero_point_from_min, quant_min_float)
    # if quant_max_float < zero_point_from_min, bool_more_quant_max_float = 1 else 0
    bool_more_quant_max_float = less_compare_float32(quant_max_float, zero_point_from_min)

    # according to above bool param to select effective value
    less_quant_min_float = topi.multiply(quant_min_float, bool_less_quant_min_float)
    more_quant_max_float = topi.multiply(quant_max_float, bool_more_quant_max_float)

    # compute which num is not less than quant_min_float and not large than quant_max_float
    tensor_one = topi.full(min_broadcast.shape, dtype, dc.one_const(dtype))
    bool_not_less_quant_min_float = topi.subtract(tensor_one, bool_less_quant_min_float)
    bool_not_more_quant_max_float = topi.subtract(tensor_one, bool_more_quant_max_float)
    bool_between_min_max = topi.multiply(bool_not_less_quant_min_float, bool_not_more_quant_max_float)
    between_min_max_float = topi.multiply(zero_point_from_min, bool_between_min_max)
    # add 0.5 to num which min <= num <= max and then floor them.
    between_min_max_add_half_one = topi.add(between_min_max_float, dc.half_const(dtype))
    between_min_max_round = akg.lang.cce.floor(between_min_max_add_half_one)
    if utils.product_is_mini():
        between_min_max_round = topi.cast(between_min_max_round, "float16")

    between_min_max_round = topi.cast(between_min_max_round, "float32")

    # calculate the maximum and minimum values of the quantization
    nudged_zero_point_tmp = topi.add(less_quant_min_float, more_quant_max_float)
    nudged_zero_point = topi.add(nudged_zero_point_tmp, between_min_max_round)

    nudged_min_tmp = topi.subtract(quant_min_float, nudged_zero_point)
    nudged_max_tmp = topi.subtract(quant_max_float, nudged_zero_point)
    nudged_min = topi.multiply(nudged_min_tmp, scale)
    nudged_max = topi.multiply(nudged_max_tmp, scale)
    res = [nudged_min, nudged_max, scale]

    return res


def bool_both_zero_compute(juduged_min, juduged_max):
    """if input min and max are both zero then output_data will be all zero,so need a juduge compute tensor"""
    dtype = juduged_min.dtype
    tensor_zero = topi.full(juduged_min.shape, dtype, dc.zero_const(dtype))
    min_abs = topi.abs(juduged_min)
    max_abs = topi.abs(juduged_max)
    min_max_replace = topi.add(min_abs, max_abs)
    # just check wether min and max are all zero, if true  return 0
    bool_min_max_product_less_zero = less_compare_float32(min_max_replace, tensor_zero)
    bool_min_max_product_more_zero = less_compare_float32(tensor_zero, min_max_replace)
    bool_both_zero = topi.add(bool_min_max_product_less_zero, bool_min_max_product_more_zero)

    return bool_both_zero

def fake_quant_with_min_max_vars_per_channel_compute(input_data, input_min, input_max, num_bits=8, narrow_range=False):
    """fake_quant_with_min_max_vars_per_channel compute implemention"""
    shape = get_shape(input_data.shape)
    dtype = input_data.dtype
    min_broadcast = akg.lang.cce.broadcast(input_min, shape, dtype)
    max_broadcast = akg.lang.cce.broadcast(input_max, shape, dtype)
    # get nudged_min and nudged_max by nudged_min_max_compute function
    nudged_min_nudged_max = nudged_min_max_compute(min_broadcast, max_broadcast, num_bits, narrow_range)
    # transform the input between nudged_max and nudged_min
    clamped_tmp = topi.minimum(input_data, nudged_min_nudged_max[1])
    clamped = topi.maximum(clamped_tmp, nudged_min_nudged_max[0])

    # calculate the quantized and dequantized results
    clamped_shifted = topi.subtract(clamped, nudged_min_nudged_max[0])
    if utils.product_is_mini():
        clamped_shifted_div_scale = mul(clamped_shifted, reciprocal(nudged_min_nudged_max[2]))
    else:
        clamped_shifted_div_scale = div(clamped_shifted, nudged_min_nudged_max[2])
    result_tmp = topi.add(clamped_shifted_div_scale, dc.half_const(dtype))
    floor_result_tmp = akg.lang.cce.floor(result_tmp)
    if utils.product_is_mini():
        floor_result_tmp = topi.cast(floor_result_tmp, "float16")

    floor_result_tmp = topi.cast(floor_result_tmp, "float32")
    scale_product = topi.multiply(floor_result_tmp, nudged_min_nudged_max[2])
    tmp_res = topi.add(scale_product, nudged_min_nudged_max[0])
    # get bool_both_zero_value by bool_both_zero_compute function
    bool_both_zero_value = bool_both_zero_compute(min_broadcast, max_broadcast)
    res = topi.multiply(tmp_res, bool_both_zero_value)

    return res
@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (int, type(None)),
                          (bool, type(None)))
def fake_quant_with_min_max_vars_per_channel(input_data, input_min, input_max, num_bits=8, narrow_range=False):
    """
    Generate fake_quantize the input_data for every channel.

    Note:
        For input_data last dim must be equal to d. And need to satisfy: input_min <= 0 <= input_max.

    Args:
        input_data (tvm.tensor.Tensor): Tensor of type float32, shape must be equal to [b, d] or [b, h, w, d] or [d].
        input_min (tvm.tensor.Tensor): Tensor of type float32, shape must be equal to [d].
        input_max (tvm.tensor.Tensor): Tensor of type float32, shape must be equal to [d].
        num_bits (int):  The quantization bits, must be int, defaults to 8.
        narror_range (Union[bool, None]): if True, quant_min equal to 1, else 0, defaults to False.

    Returns:
        tvm.tensor.Tensor of same type and shape as input_data.
    """

    # get shape and check
    shape_inputs = get_shape(input_data)
    shape_min = get_shape(input_min)
    shape_max = get_shape(input_max)
    vc_util.elemwise_shape_check(shape_min, shape_max)
    vc_util.auto_broadcast_check(shape_min, shape_inputs)
    if shape_min[0] != shape_inputs[-1]:
        raise RuntimeError("The shapes of min,max and shape_inputs last one dimension should be same!")

    # check dtype
    vc_util.ops_dtype_check(input_data.dtype, vc_util.DtypeForDavinci.FLOAT32)
    vc_util.elemwise_dtype_check(input_min.dtype, input_max.dtype, vc_util.DtypeForDavinci.FLOAT32)
    # check num_bits range
    if num_bits > 16 or num_bits < 2:
        raise ValueError("numbits should be in range [2, 16]!")

    # get output by fake_quant_with_min_max_vars_per_channel_compute function
    res = fake_quant_with_min_max_vars_per_channel_compute(input_data, input_min, input_max, num_bits, narrow_range)
    return res
