# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function: sliceeven"""
import akg.tvm
from akg.utils import validation_check as vc_util



def sliceeven(input):
    """
    Find all even index.

    Note:
        if the index is even return this index else return 0.

    Args:
        input (tvm.tensor.Tensor): Tensor of type float16, float32, must be 1D-Tensor, real input is the input's index.

    Returns:
       tvm.tensor.Tensor, has same type and shape as input.

    """

    dtype = input.dtype
    shape = [x.value for x in input.shape]
    check_list = ["float16", "float32"]
    if not dtype in check_list:
        raise RuntimeError("sliceeven_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    vc_util.check_shape(shape)
    assert len(shape) == 1

    res = akg.tvm.compute(shape, lambda i: akg.tvm.if_then_else(
        i % 2 == 0,
        input[i], akg.tvm.const(0, input.dtype)))
    return res
