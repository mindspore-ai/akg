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

"""operator dsl function: fake_quant_with_min_max_args_gradient"""

import akg
from akg import tvm, topi
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util
from test_op.fake_quant_with_min_max_args import nudge_min_max


def _cmpare_value(input_data, nudged_min, nudged_max):
    """
    where((input_data<=nudged_max)&(x>=nudged_min),1,0)

    Args:  
        input_data (tvm.tensor.Tensor): Input data
        nudged_min (tvm.tensor.Tensor): Minimum value of comparison
        nudged_max (tvm.tensor.Tensor): Maximum value of comparison

    Returns:
        tvm.tensor.Tensor
    """
    min_value = tvm.const(2**(-126), dtype="float32")
    # (2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1
    # so min_value*max_value*max_value*max_value_one = 1
    max_value = tvm.const(2**(62), dtype="float32")
    max_value_one = tvm.const(2**(2), dtype="float32")
    data_zero = topi.multiply(input_data, 0)
    max_value_tensor = topi.add(data_zero, max_value)
    min_value_tensor = topi.add(data_zero, min_value)
    max_value_one_tensor = topi.add(data_zero, max_value_one)

    sub_tmp = topi.subtract(input_data, nudged_min)
    sub_min = topi.add(sub_tmp, min_value)
    vmax_tmp = topi.maximum(sub_min, data_zero)

    sub_tmp_max = topi.subtract(nudged_max, input_data)
    sub_max = topi.add(sub_tmp_max, min_value)
    vmin_tmp = topi.maximum(sub_max, data_zero)

    one_tmp = topi.multiply(vmax_tmp, vmin_tmp)
    one_min = topi.minimum(one_tmp, min_value_tensor)

    vmul_max_value = topi.multiply(one_min, max_value_tensor)
    vmul_max_value_one = topi.multiply(vmul_max_value, max_value_tensor)
    between_nudged_min_max = topi.multiply(vmul_max_value_one, max_value_one_tensor)

    return between_nudged_min_max


@vc_util.check_input_type(tvm.tensor.Tensor,
                          tvm.tensor.Tensor,
                          (float, int, type(None)),
                          (float, int, type(None)),
                          (int, type(None)), 
                          (bool, type(None)))
def fake_quant_with_min_max_args_gradient(input_gradients, input_data, min=-6, max=6, num_bits=8, 
                                          narrow_range=False):
    """
    Computes gradients of Fake-quantize on the 'input_data' tensor,

    output_backprops = input_gradients*(if input_data>=nudged_min and <=nudged_max 1 else 0)

    Args:
        input_gradients (tvm.tensor.Tensor): input gradients from previously operation
        input_data (tvm.tensor.Tensor): input of fake-quantize, only supports "float32"
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
    vc_util.check_shape(shape)
    vc_util.elemwise_shape_check(input_gradients.shape, input_data.shape)

    vc_util.ops_dtype_check(input_data.dtype, vc_util.DtypeForDavinci.FLOAT32)
    vc_util.ops_dtype_check(input_gradients.dtype, vc_util.DtypeForDavinci.FLOAT32)

    nudged_min, nudged_max, scale = nudge_min_max(min, max, num_bits, 
                                                   narrow_range)

    zero_tensor = tvm.compute(input_data.shape, 
                              lambda *i: tvm.const(0, dtype="float32"),
                              name="zero_tensor")
    nudged_max_tensor = topi.add(zero_tensor, nudged_max)
    nudged_min_tensor = topi.add(zero_tensor, nudged_min)
    
    # where((input_data<=nudged_max)&(x>=nudged_min),1,0),Convert the input to 0 and 1 tensor
    between_nudged_min_max = _cmpare_value(input_data, nudged_min_tensor, nudged_max_tensor)

    res = topi.multiply(input_gradients, between_nudged_min_max)

    return res
