# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: equal_count"""
import akg.topi
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini
from akg.utils.dsl_create import produce_shapes
from akg.utils.format_transform import get_shape
from ..cast import cast
from ..sum import sum


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def equal_count(x, y, target=utils.CCE):
    """
    compute equal num of x and y.

    Args:
        x (tvm.tensor.Tensor): Tensor of type int32.
        y (tvm.tensor.Tensor): Tensor of type int32.

    Returns:
        tvm.tensor.Tensor, equal num, type is int32.

    Supported Platforms:
        'Ascend'
    """
    # check shapes
    shape1 = get_shape(x)
    shape2 = get_shape(y)
    shapes = [shape1, shape2]
    for _, shape_ in enumerate(shapes):
        utils.check_shape(shape_)
    if len(shape1) != 1 or len(shape2) != 1:
        raise RuntimeError("Two inputs should all be one dim!")

    # check types
    dtype = x.dtype
    utils.ops_dtype_check([x.dtype, y.dtype], utils.DtypeForDavinci.INT32)

    # Due to instruction limitations, the int32 data needs to be converted to
    # float16 or float32.
    # When the int32 data is casted to float16, there may be overflow problems,
    # so as far as possible the int32 data is casted to float32.
    orig_dtype = dtype
    if product_is_mini():
        dtype = "float16"
    else:
        dtype = "float32"
    x = cast(x, dtype, target)
    y = cast(y, dtype, target)

    shape1, shape2, shape = produce_shapes(shape1, shape2)
    t = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(1, dtype), "t")
    f = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(0, dtype), "f")
    x = akg.topi.broadcast_to(x, shape)
    y = akg.topi.broadcast_to(y, shape)
    z = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(
                        x[indice] == y[indice], t[indice], f[indice]), name="z")
    res = sum(z, target=target)
    if res.dtype != orig_dtype:
        res = cast(res, orig_dtype, target)
    return res
