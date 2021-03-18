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

"""operator dsl function: divide"""
import akg.topi
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def divide(lhs, rhs):
    """
    Calculate divide.

    Args:
        lhs: The left tensor.
        rhs: The right tensor.

    Returns:
        tvm.tensor.Tensor.
    """
    shape_l = [x.value for x in lhs.shape]
    shape_r = [x.value for x in rhs.shape]
    vc_util.check_shape(shape_l)
    vc_util.check_shape(shape_r)
    vc_util.auto_broadcast_check(shape_l, shape_r)
    vc_util.elemwise_dtype_check(lhs.dtype, rhs.dtype)
    output = akg.topi.divide(lhs, rhs)

    return output
