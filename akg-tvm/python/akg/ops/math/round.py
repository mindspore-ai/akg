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

"""operator dsl function: round"""
import akg
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def round_(data, target=utils.CCE):
    """
    Round elements of x to nearest integer.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32, int8, unit8, int32.

    Returns:
        tvm.tensor.Tensor of same type and shape as data.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    utils.check_shape(data.shape)
    in_type = data.dtype
    if target == utils.CCE:
        if in_type != 'float16':
            data = akg.tvm.compute(data.shape, lambda *i: data(*i).astype("float16"), name="data_f16")
        return akg.lang.ascend.round(data)
    if in_type == 'float16':
        data = akg.topi.cast(data, 'float32')
    output = akg.topi.round(data)
    if in_type == 'float16':
        output = akg.topi.cast(output, 'float16')
    return output
