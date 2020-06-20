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

"""operator dsl function: cross"""
import akg
from akg import tvm, topi
from akg.utils.format_transform import get_shape

from akg.utils import validation_check as vc_util, kernel_exec as utils
from akg.ops.math.cast import cast

@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def cross(x, y):
    """
    Compute cross product of x and y.

    Note:
        The first dim of x or y must be 3, it will be calculated as (two dims for example)
        .. math::
            res = x \\times y = \\left[ \\begin{matrix}
            l, & \\cdots \\\\ m, & \\cdots \\\\ n, & \\cdots
            \\end{matrix} \\right] \\times \\left[ \\begin{matrix}
            o, & \\cdots \\\\ p, & \\cdots \\\\ q, & \\cdots
            \\end{matrix} \\right] = \\left[ \\begin{matrix}
            mq-np, & \\cdots \\\\ no-lq, & \\cdots \\\\ lp-mo, & \\cdots \\\\
            \\end{matrix} \\right]

    Args:
        x (tvm.tensor.Tensor): Input tensor, only support float16, float32,
                               int32, int8, uint8.
        y (tvm.tensor.Tensor): Input tensor, must have the same shape and dtype
                               as x.

    Returns:
        A tvm.tensor.Tensor with the same shape and dtype as x.
    """
    vc_util.elemwise_shape_check(get_shape(y), get_shape(x))
    vc_util.elemwise_dtype_check(
        y.dtype, x.dtype,
        (vc_util.DtypeForDavinci.ALL_FLOAT) if utils.product_is_mini() \
            else (vc_util.DtypeForDavinci.FLOAT16,
                  vc_util.DtypeForDavinci.FLOAT32,
                  vc_util.DtypeForDavinci.INT32,
                  vc_util.DtypeForDavinci.INT8, vc_util.DtypeForDavinci.UINT8))

    shape = get_shape(x)

    if shape[0] != 3:
        raise RuntimeError(
            "The first axis of input must be 3, actual input is %d" % shape[0])

    inp_dtype = x.dtype
    need_type_convert = inp_dtype in ("int8", "uint8")

    shape = get_shape(x)
    shp = shape[1:]

    if need_type_convert:
        x = cast(x, "float16")
        y = cast(y, "float16")

    a0b1 = tvm.compute(shp, lambda *i: x(0, *i) * y(1, *i), name="a0b1")
    a0b2 = tvm.compute(shp, lambda *i: x(0, *i) * y(2, *i), name="a0b2")
    a1b0 = tvm.compute(shp, lambda *i: x(1, *i) * y(0, *i), name="a1b0")
    a1b2 = tvm.compute(shp, lambda *i: x(1, *i) * y(2, *i), name="a1b2")
    a2b0 = tvm.compute(shp, lambda *i: x(2, *i) * y(0, *i), name="a2b0")
    a2b1 = tvm.compute(shp, lambda *i: x(2, *i) * y(1, *i), name="a2b1")

    res0 = tvm.compute(shp, lambda *i: a1b2(*i) - a2b1(*i), name="res0")
    res1 = tvm.compute(shp, lambda *i: a2b0(*i) - a0b2(*i), name="res1")
    res2 = tvm.compute(shp, lambda *i: a0b1(*i) - a1b0(*i), name="res2")

    res = tvm.compute(
        shape,
        lambda *i:
        tvm.expr.Select(
            i[0] == 0,
            res0(*i[1:]),
            tvm.expr.Select(i[0] == 1, res1(*i[1:]), res2(*i[1:]))),
        name='res')

    if need_type_convert:
        res = cast(res, inp_dtype)

    return res
