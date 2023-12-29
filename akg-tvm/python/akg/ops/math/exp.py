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

"""operator dsl function: exp"""
import akg.topi
import akg.tvm
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini


@utils.check_input_type(akg.tvm.tensor.Tensor)
def _exp(data):
    shape = [x.value for x in data.shape]
    utils.check_shape(shape)
    output = akg.topi.exp(data)

    return output


@utils.check_input_type(akg.tvm.tensor.Tensor)
def _exp_ascend(in_data):
    dtype = in_data.dtype
    utils.check_shape(in_data.shape)
    if dtype == "float32" and product_is_mini():
        in_data = akg.tvm.compute(in_data.shape, lambda *indice: in_data(*indice).astype("float16"), name='type_cast')

    output = akg.tvm.compute(in_data.shape, lambda *index: akg.tvm.exp(in_data(*index)), name='exp')

    if dtype == "float32" and product_is_mini():
        output = akg.tvm.compute(in_data.shape, lambda *indice: output(*indice).astype("float32"), name='res')

    return output


def exp(data, target=utils.CCE):
    """
    Calculate exponential of input data.

    Args:
        input (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor, has the same type as input.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return _exp_ascend(data)
    else:
        return _exp(data)