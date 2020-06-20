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

"""operator dsl function: reduce_all"""

import akg.topi
import akg.tvm
import akg
from akg.utils import validation_check as vc_util
from akg.utils import format_transform as ft_util
from akg.utils import dsl_create as dc


@vc_util.check_input_type(akg.tvm.tensor.Tensor, (int, list, tuple, type(None)), (bool, type(None)))
def reduce_all(data, axis=None, keepdims=False):
    """
    Computes logical and of the input tensor.

    Args:
        data(tvm.tensor.Tensor): Tensor of type Boolean.
        axis(Union[None, int, list, tuple]): Specifies which axes to reduce, if None, all dimensions of
            input tensor data will be reduced and the shape of output tensor will be (1,).
        keepdims(Union[None, bool]): if true, keep the dimensions with length 1.

    Returns:
        tvm.tensor.Tensor of same type as input tensor data.
    """

    shape = [x.value for x in data.shape]

    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.BOOL)
    vc_util.check_shape(shape)

    if axis is None and keepdims is False:
        raise ValueError("keepdims must be True when axis is None!")

    axis_new = ft_util.refine_reduce_axis(data, axis)

    xx1 = akg.tvm.compute(shape, lambda *indice: data(*indice).astype("float16"), name='xx1')
    xx = (-xx1 + dc.one_const("float16"))
    yy = akg.topi.sum(xx, axis=axis_new, keepdims=keepdims)

    o_shape = list(yy.shape)

    zz = akg.tvm.compute(o_shape, lambda *indice: yy(*indice).astype("bool"), name='zz')

    y1 = akg.tvm.compute(o_shape, lambda *indice: akg.tvm.expr.Select(zz(*indice), dc.zero_const("float16"), dc.one_const("float16")), name="y1")

    y = akg.tvm.compute(o_shape, lambda *indice: y1(*indice).astype("bool"), name='y')

    return y
