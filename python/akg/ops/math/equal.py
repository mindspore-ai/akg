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

"""operator dsl function: equal"""
import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils.dsl_create import produce_shapes
from akg.utils.kernel_exec import product_is_mini
from .cast import cast
from .sub import sub


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def _equal(input1, input2):

    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)

    shape1, shape2, shape = produce_shapes(shape1, shape2)

    utils.elemwise_dtype_check(input1.dtype, input2.dtype)
    dtype = input1.dtype

    # get equal compute
    t_value = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(1, dtype), "T")
    f_value = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(0, dtype), "F")

    input1_bro = akg.topi.broadcast_to(input1, shape)
    input2_bro = akg.topi.broadcast_to(input2, shape)
    c_out = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(input1_bro[indice] == input2_bro[indice],
                                                                         t_value[indice], f_value[indice]), name="C")
    res = akg.tvm.compute(shape, lambda *indice: c_out(*indice).astype("bool"), name="res")

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def _equal_ascend(input1, input2, target=utils.CCE):
    # check shapes
    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    shapes = [shape1, shape2]
    for _, shp in enumerate(shapes):
        utils.check_shape(shp)

    utils.ops_dtype_check([input1.dtype, input2.dtype],
                            [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32,
                             utils.DtypeForDavinci.INT8, utils.DtypeForDavinci.UINT8])

    dtype = input1.dtype
    orig_dtype = dtype
    if product_is_mini() and dtype != "float16":
        dtype = "float16"
    if (not product_is_mini()) and dtype not in ("float16", "float32"):
        # for int32, if cast to float16, there may be overflow
        dtype = "float32"

    if orig_dtype == "float32" and dtype == "float16":
        input_sub = sub(input1, input2, target)
        input_sub = cast(input_sub, dtype, target)
        zero = akg.tvm.const(0.0, dtype)
        res = akg.topi.equal(input_sub, zero)
    else:
        input1 = cast(input1, dtype, target)
        input2 = cast(input2, dtype, target)
        res = akg.topi.equal(input1, input2)
    return res


def Equal(input1, input2, target=utils.CCE):
    """
    check whether input1 equals to input2.

    Args:
        input1 (tvm.tensor.Tensor): Tensor.
        input2 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor. If input1 equal to input2 return True, else return False.
    
    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return _equal_ascend(input1, input2)
    else:
        return _equal(input1, input2)