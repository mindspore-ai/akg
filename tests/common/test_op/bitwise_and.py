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

"""operator dsl function: bitwise_and"""
import akg
from akg import tvm, topi
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import produce_shapes


def _check_parameters(x1, x2):
    """check the input parameters"""
    shape_x = get_shape(x1)
    shape_y = get_shape(x2)
    dtype_x = x1.dtype
    dtype_y = x2.dtype

    vc_util.check_shape(shape_x)
    vc_util.check_shape(shape_y)

    vc_util.ops_dtype_check(
        [dtype_x, dtype_y],
        [vc_util.DtypeForDavinci.INT16, vc_util.DtypeForDavinci.UINT16])
    if dtype_x != dtype_y:
        raise RuntimeError(
            "two input type must be the same, type of x is %s, type of y is %s" % (
                dtype_x, dtype_y))


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def bitwise_and(x1, x2):
    """
    Computes the bitwise and of `x1` and `x2`.

    Args:
        x1 (tvm.tensor.Tensor): tensor x1, only support int16,uint16.
        x2 (tvm.tensor.Tensor): tensor x2, only support int16,uint16.

    Returns:
        A tvm.tensor.Tensor as result of bitwise and.
    """
    _check_parameters(x1, x2)

    shape_x = get_shape(x1)
    shape_y = get_shape(x2)
    _, _, shape_max = produce_shapes(shape_x, shape_y)

    data_x = topi.broadcast_to(x1, shape_max)
    data_y = topi.broadcast_to(x2, shape_max)

    res = tvm.compute(data_x.shape,
                      lambda *i: data_x(*i) & data_y(*i),
                      name="and_res")

    return res
