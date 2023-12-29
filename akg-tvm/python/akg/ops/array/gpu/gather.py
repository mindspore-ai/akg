# Copyright 2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: scatter_add"""
import akg.tvm
import akg.utils as  utils
from akg.composite import gather as cuda_gather


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, int)
def gather(data, indices, axis):
    """
    Supported Platforms:
        'GPU'
    """
    dim = data.ndim
    if axis < -dim or axis >= dim:
        raise ValueError(f'axis {axis} is out of bounds for array with dim {dim}')
    axis = axis % dim
    return cuda_gather((data, indices), {'axis': [axis]})
