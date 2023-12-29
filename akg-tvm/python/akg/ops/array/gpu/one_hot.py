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

"""operator dsl function: one_hot"""
import akg.tvm
import akg.topi
import akg.utils as  utils


@utils.check_input_type(akg.tvm.tensor.Tensor, int, int, int, int, str)
def one_hot(indices, on_value, off_value, depth, axis, dtype):
    """
    Supported Platforms:
        'GPU'
    """
    on_value_const = akg.tvm.const(on_value, dtype)
    off_value_const = akg.tvm.const(off_value, dtype)
    output = akg.topi.one_hot(indices, on_value_const, off_value_const, depth, axis, dtype)
    return output
