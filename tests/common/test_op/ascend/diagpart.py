# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function:diagpart"""

import akg.tvm
import akg.utils as utils

def diagpart(data, target=utils.CCE):
    """
    Returns the diagonal part of data.

    Args:
        data: Tensor.

    Returns:
        Tensor, has the same type as data and shape of data.shape[0:len-1].
    """

    shape = [x.value for x in data.shape]
    utils.check_shape(shape)
    rank = len(shape)
    if rank not in (2, 4, 6, 8):
        raise ValueError("diagpart only support even rank (2/4/6/8) while the rank is {}".format(rank))

    o_shape = []
    for i in range(rank // 2):
        if shape[i] == shape[rank // 2 + i]:
            o_shape.append(shape[i])
        else:
            raise ValueError("diagpart only support square matrix while the shape is {}".format(shape))

    dtype = data.dtype
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])

    if rank == 2:
        res = akg.tvm.compute(o_shape, lambda i: data[i, i])
    elif rank == 4:
        res = akg.tvm.compute(o_shape, lambda i, j: data[i, j, i, j])
    elif rank == 6:
        res = akg.tvm.compute(o_shape, lambda i, j, m: data[i, j, m, i, j, m])
    elif rank == 8:
        res = akg.tvm.compute(o_shape, lambda i, j, m, n: data[i, j, m, n, i, j, m, n])
    return res
