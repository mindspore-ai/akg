# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: abs"""
import akg.topi
import akg.tvm
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def abs(in_data, target=utils.CCE):
    """
    Compute absolute value of a tensor.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32, int8, unit8, int32.

    Returns:
        tvm.tensor.Tensor of same type and shape as data.
    
    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    utils.check_shape(in_data.shape)
    in_type = in_data.dtype
    if target == utils.CCE:
        utils.ops_dtype_check(in_type, utils.DtypeForDavinci.ALL_TYPES)
        need_cast_dtype = ["int8", "int32", "uint8"]
        if in_type in need_cast_dtype:
            in_data = akg.tvm.compute(in_data.shape, lambda *indice: in_data(*indice).astype("float16"), name='type_cast')
        output = akg.tvm.compute(in_data.shape, lambda *index: akg.tvm.abs(in_data(*index)), name='abs_value')
        if in_type in need_cast_dtype:
            output = akg.tvm.compute(in_data.shape, lambda *indice: output(*indice).astype(in_type), name='res')
    else:
        if in_type == 'float16':
            in_data = akg.topi.cast(in_data, 'float32')
        output = akg.topi.abs(in_data)
        if in_type == 'float16':
            output = akg.topi.cast(output, 'float16')
    return output