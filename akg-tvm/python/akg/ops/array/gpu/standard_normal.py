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

"""operator dsl function: standard_normal"""
import akg.tvm
import akg.utils as  utils
from akg.composite import standard_normal as cuda_standard_normal


@utils.check_input_type(int, tuple)
def standard_normal(seed, shape):
    """
    Operator dsl function for standard_normal.

    Args:
        seed (int): Random seed.
        shape (tuple(int)): Output shape.

    Returns:
        Tensor with the given shape.

    Supported Platforms:
        'GPU'
    """
    return cuda_standard_normal(None, {"seed": seed, "shape": shape})
