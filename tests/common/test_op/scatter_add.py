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

"""operator dsl function: scatter_add"""
from functools import reduce

import akg
from akg import tvm, topi
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.tvm.hybrid import script
from akg.utils.dsl_create import TensorUtils

attrs = {
    "RewriteVarTensorIdx": True
}



@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def scatter_add(ref, indices, updates):
    """
    Add ref with updates based on sparse index: indices.

    Note:
        updates.shape need equal to indices.shape + ref.shape[1:].

    Args:
        ref (tvm.tensor.Tensor): Tensor of type float16, float32, int8, int32 and uint8.
        indices (tvm.tensor.Tensor): Tensor of type int32.
        updates (tvm.tensor.Tensor): Tensor has the same type as ref.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as ref.

    """
    shape_ref = get_shape(ref)
    shape_indices = get_shape(indices)
    shape_updates = get_shape(updates)

    vc_util.check_shape(shape_ref)
    vc_util.check_shape(shape_indices)
    vc_util.check_shape(shape_updates)
    vc_util.ops_dtype_check([ref.dtype, updates.dtype], [vc_util.DtypeForDavinci.ALL_FLOAT,
                                                         vc_util.DtypeForDavinci.INT32])

    vc_util.ops_dtype_check(indices.dtype, vc_util.DtypeForDavinci.INT32)
    new_shape_indices = (reduce(lambda x, y: x * y, shape_indices), )
    if len(shape_ref) > 1:
        new_shape_ref = (shape_ref[0], reduce(lambda x, y: x * y, shape_ref[1:]))
        new_indices = topi.reshape(indices, new_shape_indices)
        new_updates_shape = (tuple(new_indices.shape) + tuple(new_shape_ref[1:]))
        new_updates = topi.reshape(updates, new_updates_shape)
        new_ref = topi.reshape(ref, new_shape_ref)
    else:
        new_indices = topi.reshape(indices, new_shape_indices)
        new_updates_shape = (tuple(new_indices.shape) + tuple(shape_ref[1:]))
        new_updates = topi.reshape(updates, new_updates_shape)
        new_ref = ref

    # 1D case hybrid
    @script
    def scatter_add_1d(input, input_indices, input_updates):
        n, = input.shape
        idx_len = input_indices.shape[0]
        for i in range(n):
            for idx in range(idx_len):
                if i == input_indices[idx]:
                    input[input_indices[idx]] += input_updates[idx]
        return input

    # ND case reshape to 2D's hybrid, now 2D -- 5D are OK
    @script
    def scatter_add(input, input_indices, input_updates):
        n, h = input.shape
        idx_len = input_indices.shape[0]
        for i in range(n):
            for idx in range(idx_len):
                if i == input_indices[idx]:
                    for j in range(h):
                            input[input_indices[idx], j] += input_updates[idx, j]
        return input

    if len(shape_ref) == 1:
        out = scatter_add_1d(new_ref, new_indices, new_updates)
    else:
        out = scatter_add(new_ref, new_indices, new_updates)
        out = topi.reshape(out, shape_ref)
        attrs["enable_feature_library"] = True
    out, binds_info = TensorUtils.inplace_set(ref, out)
    attrs[utils.BINDS] = binds_info

    return out, attrs
