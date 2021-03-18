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

"""operator dsl function: fake_quant_with_min_max_vars_per_channel_gradient"""

import akg
from akg import tvm, topi
from akg.utils import dsl_create as dc
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util
from tests.common.test_op.fake_quant_with_min_max_vars_per_channel import bool_both_zero_compute
from tests.common.test_op.fake_quant_with_min_max_vars_per_channel import nudged_min_max_compute

def _less_equal_compare_float32(data_x, data_y):
    """if x <= y, then return 1, else 0"""
    data_out = tvm.compute(data_x.shape, lambda *index: tvm.expr.Select(data_x(*index) <= data_y(*index),
                                                                        dc.one_const(data_x.dtype),
                                                                        dc.zero_const(data_x.dtype)))
    return data_out

def _bool_negate(input_bool):
    """Negate every value"""
    return topi.subtract(dc.one_const(input_bool.dtype), input_bool)

def fake_quant_with_min_max_vars_per_channel_gradient_compute(input_gradients, inputs_data,
                                                              min_broadcast, max_broadcast,
                                                              num_bits=8,
                                                              narrow_range=False):
    """Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation."""
    shape = get_shape(inputs_data)
    sum_axis = [x for x in range(0, len(shape) - 1)]
    dtype = inputs_data.dtype

    nudged_min, nudged_max, _ = nudged_min_max_compute(min_broadcast, max_broadcast, num_bits, narrow_range)
    # both zero yields zero
    bool_both_zero_value = bool_both_zero_compute(min_broadcast, max_broadcast)
    bool_both_zero_negate = _bool_negate(bool_both_zero_value)

    bool_less_equal_nudged_max = _less_equal_compare_float32(inputs_data, nudged_max)
    bool_more_equal_nudged_min = _less_equal_compare_float32(nudged_min, inputs_data)
    bool_between_nudged_min_max = topi.multiply(bool_less_equal_nudged_max, bool_more_equal_nudged_min)
    # gradient is 1 if input in [min, max] else 0
    backprops_input_tmp = topi.multiply(bool_between_nudged_min_max, input_gradients)
    backprops_bool_both_zero = topi.multiply(backprops_input_tmp, bool_both_zero_value)
    # if min and max are both zero, gradients is input_gradients
    input_gradients_both_zero = topi.multiply(input_gradients, bool_both_zero_negate)
    backprops_input = topi.add(backprops_bool_both_zero, input_gradients_both_zero)

    # gradients for min is input_gradients if inputs_data < nudged_min else 0
    bool_less_nudged_min = _bool_negate(bool_more_equal_nudged_min)
    output_backprop_min_tmp = topi.multiply(bool_less_nudged_min, input_gradients)
    # gradients for min is 0 if min and max are both 0
    output_backprop_min_bool = topi.multiply(output_backprop_min_tmp, bool_both_zero_value)
    if sum_axis == []:
        output_backprop_min = output_backprop_min_bool
    else:
        output_backprop_min = topi.sum(output_backprop_min_bool, sum_axis)

    # gradients for max is input_gradients if inputs_data > nudged_max else 0
    bool_more_nudged_max = _bool_negate(bool_less_equal_nudged_max)
    output_backprop_max_tmp = topi.multiply(bool_more_nudged_max, input_gradients)
    # gradients for max is 0 if min and max are both 0
    output_backprop_max_bool = topi.multiply(output_backprop_max_tmp, bool_both_zero_value)
    if sum_axis == []:
        output_backprop_max = output_backprop_max_bool
    else:
        output_backprop_max = topi.sum(output_backprop_max_bool, sum_axis)
    return backprops_input, output_backprop_min, output_backprop_max

@vc_util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor,
                          tvm.tensor.Tensor, tvm.tensor.Tensor,
                          (int, type(None)), (bool, type(None)))
def fake_quant_with_min_max_vars_per_channel_gradient(input_gradients, input_data, 
                                                      input_min, input_max,
                                                      num_bits=8, narrow_range=False):
    """
    Computes gradients of Fake-quantize on the 'input_data' tensor,

    output_backprops = input_gradients*(if input_data>=nudged_min and <=nudged_max 1 else 0)

    Args:
        input_gradients (tvm.tensor.Tensor): input gradients from previously operation
        input_data (tvm.tensor.Tensor): input of fake-quantize, only supports "float32"
        input_min (tvm.tensor.Tensor): input_min shape equals to input_max shape
            The last dimension shoud be same for shapes of min, max and shape_inputs
            only support fp32
        input_max (tvm.tensor.Tensor): only support fp32
        num_bits (int): Defaults to 8. bitwidth of the quantization,between 2 and 16
        narrow_range (bool): 
            True, quantized into the quantization range [1, 2^num_bits - 1]
            False,quantized into the quantization range [0, 2^num_bits - 1]

    Returns:
        tvm.tensor.Tensor
    """
    input_gradients_shape = get_shape(input_gradients)
    input_data_shape = get_shape(input_data)
    input_min_shape = get_shape(input_min)
    input_max_shape = get_shape(input_max)

    vc_util.check_shape(input_gradients_shape)
    vc_util.check_shape(input_data_shape)
    vc_util.check_shape(input_min_shape)
    vc_util.check_shape(input_max_shape)

    vc_util.elemwise_shape_check(input_gradients.shape, input_data.shape)
    vc_util.elemwise_shape_check(input_min_shape, input_max_shape)
    if input_min_shape[0] != input_data_shape[-1]:
        raise RuntimeError(
            "The shapes of min,max and shape_inputs last one dimension shoud be same")

    vc_util.ops_dtype_check(input_gradients.dtype, vc_util.DtypeForDavinci.FLOAT32)
    vc_util.ops_dtype_check(input_data.dtype, vc_util.DtypeForDavinci.FLOAT32)
    vc_util.ops_dtype_check(input_min.dtype, vc_util.DtypeForDavinci.FLOAT32)
    vc_util.ops_dtype_check(input_max.dtype, vc_util.DtypeForDavinci.FLOAT32)

    if num_bits > 16 or num_bits < 2:
        raise RuntimeError("numbits should be range[2,16]")

    input_min_broadcast = topi.broadcast_to(input_min, input_data_shape)
    input_max_broadcast = topi.broadcast_to(input_max, input_data_shape)

    res = fake_quant_with_min_max_vars_per_channel_gradient_compute(input_gradients, input_data,
                                                                    input_min_broadcast,
                                                                    input_max_broadcast,
                                                                    num_bits,
                                                                    narrow_range)
    return res
