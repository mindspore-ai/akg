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

"""operator dsl function: scatter_nd"""

import akg.tvm
from akg.utils import validation_check as vc_util


def scatter_nd(indices, updates, shape):
    """
    Scatters input tensor updates to a new tensor according to indices.

    Args:
        indices(akg.tvm.Tensor): Tensor of type int32.
        updates(akg.tvm.Tensor): Tensor of type float16, float32, int32.
        shape(list, tuple): Specifies the shape of output tensor.

    Returns:
        Scattered tensor with same type as input tensor updates and shape specified by parameter shape.
    """

    # check shapes dtype
    indices_shape = [x.value for x in indices.shape]
    data_shape = [x.value for x in updates.shape]

    vc_util.check_shape(indices_shape)
    vc_util.check_shape(data_shape)

    indices_dtype = indices.dtype
    if not indices_dtype in "int32":
        raise TypeError("indices_dtype only support int32 while dtype is %s" % indices_dtype)
    dtype = updates.dtype
    support_list = {"float16", "float32", "int32"}
    if not (dtype in support_list):
        raise TypeError("scatter_nd only support %s while dtype is %s" % (",".join(support_list), dtype))

    n = indices.shape[0].value

    def pick(i, j, *indexes):
        return akg.tvm.expr.Select(j == indices[i][0],
                               akg.tvm.const(1, updates.dtype),
                               akg.tvm.const(0, updates.dtype)) * updates[(i,) + indexes]

    reducible = akg.tvm.compute([n] + list(shape), lambda *i: pick(i[0], i[1], *i[2:]), name="reduc")
    k = akg.tvm.reduce_axis((0, n))
    res = akg.tvm.compute(shape, lambda *i: akg.tvm.sum(reducible[(k,) + i], axis=k))
    return res
