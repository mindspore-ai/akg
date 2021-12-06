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

"""operator dsl function: bitwise_or"""
import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.utils.dsl_create import produce_shapes

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def bitwise_or(x1, x2, target=utils.CCE):
    """
    Computes the bitwise or of `x1` and `x2`.

    Args:
        x1 (tvm.tensor.Tensor): Tensor of type int16, uint16.
        x2 (tvm.tensor.Tensor): Tensor of type int16, uint16.

    Returns:
        tvm.tensor.Tensor, has the same type as x1.
    """
    # check shape
    utils.check_shape(x1)
    utils.check_shape(x2)
    _, _, output_shape = produce_shapes(get_shape(x1), get_shape(x2))

    # check input tensor data_type
    utils.ops_dtype_check([x1.dtype, x2.dtype], [utils.DtypeForDavinci.INT16, utils.DtypeForDavinci.UINT16])
    dtype = x1.dtype
    if dtype != x2.dtype:
        raise RuntimeError("input type must be same, but got %s  vs %s",
                           dtype, x2.dtype)
    
    x1 = akg.topi.broadcast_to(x1, output_shape)
    x2 = akg.topi.broadcast_to(x2, output_shape)
    res = akg.tvm.compute(output_shape, lambda *indice: x1(*indice) | x2(*indice))
    return res
