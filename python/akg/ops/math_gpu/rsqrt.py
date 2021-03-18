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

"""operator dsl function: rsqrt"""
import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def rsqrt(data1):
    """
    Computes data1 elementwise.

    Args:
        data1 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor, inverse sqaure root of data1, with same type as input tensors.
    """
    # vc_util.elemwise_dtype_check(data1.dtype)
    vc_util.ops_dtype_check(data1.dtype, ["float32", "float16"])
    vc_util.check_shape(data1.shape)

    res = akg.topi.rsqrt(data1)

    return res
