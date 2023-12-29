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

"""operator dsl function:squeeze"""

import akg.topi
import akg.utils as utils
from akg.utils.format_transform import get_shape


def squeeze(data, axis, target="cce"):
    """
    Remove the dimensions which have shape size 1.

    Args:
        data: Tensor, input whose shape is to be squeeze.
        axis: Interger, specify which size 1 dimension to be removed.

    Return:
        Tensor, has the same type and element as data, but some size 1 dimensions are removed.
    """
    shape = get_shape(data)
    if len(shape) == 1:
        raise RuntimeError("invalid input shape")
    utils.check_shape(shape)
    utils.ops_dtype_check(data.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])
    new_shape = []
    shape_to_squeeze = []
    if axis is None:
        axis = [i for i, sh in enumerate(shape) if sh == 1]
    if not isinstance(axis, (list, tuple)):
        axis = [axis]
    for i, sh in enumerate(shape):
        if not isinstance(sh, int) or i not in axis:
            new_shape.append(sh)
            shape_to_squeeze.append(True)
        else:
            shape_to_squeeze.append(False)

    def get_old_indices(indices):
        old_indices = []
        new_index = 0
        for i, sh in enumerate(shape_to_squeeze):
            if sh:
                old_indices.append(indices[new_index])
                new_index += 1
            else:
                old_indices.append(0)
        return old_indices
    
    B = akg.tvm.compute(new_shape, lambda *indices: data(*get_old_indices(indices)))
    return B
