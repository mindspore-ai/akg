#!/usr/bin/env python3
# coding: utf-8
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

"""operator dsl function:one hot"""
import akg.tvm
import akg.utils as utils
from akg.tvm.hybrid import script
from akg.utils import custom_tiling as ct_util

def onehot_tiling_strategy(tensor, axis):
    """Custom tiling strategy for onehot op."""
    tot_axis = ct_util.create_constraint_on_tensor(tensor=tensor,
                                                   values=0,
                                                   constraints=ct_util.TileConstraint.SET_PRIORITY,
                                                   tensor_pos=axis)
    return tot_axis


@utils.check_input_type(akg.tvm.tensor.Tensor, int, str, (int, float, type(None)),
                  (int, float, type(None)), (int, type(None)), (str, type(None)))
def OneHot(indices, depth, dtype, on_value=None, off_value=None, axis=None, target=utils.CCE):
    """
    generate the one-hot code for input indices

    Args:
        indices (tvm.tensor.Tensor): defining the input data.
        depth (int): defining the depth of the one hot dimension.
        dtype (String): "float16" or "float32" or "int" or "int32".
        on_value (Scalar): optional. defining the value to fill in the output if indices[i] == j. default 1.
        off_value (Scalar): optional. defining the value to fill in the output if indices[i] != j. default 0.
        axis (int): optional. The axis to fill. default -1, that means a new inner-most axis.
        attrs (dict): optional.  Dictionary provide tiling information for poly.
        kernel_name (String): optional. the name of the kernel that will be generated.

    Returns:
        akg.tvm.module. A module that combines both host and device code.
    
    Supported Platforms:
        'Ascend'
    """

    utils.ops_dtype_check([indices.dtype, dtype], utils.DtypeForDavinci.INT32.value + utils.DtypeForDavinci.ALL_FLOAT.value)

    shape = [x.value for x in indices.shape]
    utils.check_shape(shape)

    # Tensor of tensor do not support tensor with more than 3 dimensions for now
    if len(shape) > 3:
        raise RuntimeError("one_hot do not support input shape %d dimensions which is more than 3" % len(shape))

    on_value_const = akg.tvm.const(1, dtype) if on_value is None else akg.tvm.const(on_value, dtype)
    off_value_const = akg.tvm.const(0, dtype) if off_value is None else akg.tvm.const(off_value, dtype)

    if axis is None:
        axis = -1

    if axis == -1:
        axis = len(shape)

    if axis <= -2 or axis > len(shape):
        raise RuntimeError("axis(%s) is not an valid index" % axis)

    in_shape = [x for x in indices.shape]

    in_shape.insert(axis, depth)
    out_shape = tuple(in_shape)

    @script
    def one_hot_hybrid_1(indices_in, on_value_const_in, off_value_const_in):
        out = output_tensor(out_shape, on_value_const_in.dtype)

        m, n = out_shape

        for i in range(m):
            for j in range(n):
                out[i, j] = off_value_const_in

        if axis == 0:
            for i in range(n):
                if indices_in[i] >= 0:
                    out[indices_in[i], i] = on_value_const_in
        else:
            for i in range(m):
                if indices_in[i] >= 0:
                    out[i, indices_in[i]] = on_value_const_in

        return out

    @script
    def one_hot_hybrid_2(indices_in, on_value_const_in, off_value_const_in):
        out = output_tensor(out_shape, on_value_const_in.dtype)

        m, n, k = out.shape

        for x in range(m):
            for y in range(n):
                for z in range(k):
                    out[x, y, z] = off_value_const_in

        if axis == 0:
            for i in range(n):
                for j in range(k):
                    if indices_in[i, j] >= 0:
                        out[indices_in[i, j], i, j] = on_value_const_in
        elif axis == 1:
            for i in range(m):
                for j in range(k):
                    if indices_in[i, j] >= 0:
                        out[i, indices_in[i, j], j] = on_value_const_in
        else:
            for i in range(m):
                for j in range(n):
                    if indices_in[i, j] >= 0:
                        out[i, j, indices_in[i, j]] = on_value_const_in

        return out

    @script
    def one_hot_hybrid_3(indices_in, on_value_const_in, off_value_const_in):
        out = output_tensor(out_shape, on_value_const_in.dtype)
        m, n, k, t = out.shape

        for x in range(m):
            for y in range(n):
                for z in range(k):
                    for u in range(t):
                        out[x, y, z, u] = off_value_const_in

        if axis == 0:
            for i in range(n):
                for j in range(k):
                    for c in range(t):
                        if indices_in[i, j, c] >= 0:
                            out[indices_in[i, j, c], i, j, c] = on_value_const_in
        elif axis == 1:
            for i in range(m):
                for j in range(k):
                    for c in range(t):
                        if indices_in[i, j, c] >= 0:
                            out[i, indices_in[i, j, c], j, c] = on_value_const_in
        elif axis == 2:
            for i in range(m):
                for j in range(n):
                    for c in range(t):
                        if indices_in[i, j, c] >= 0:
                            out[i, j, indices_in[i, j, c], c] = on_value_const_in
        else:
            for i in range(m):
                for j in range(n):
                    for c in range(k):
                        if indices_in[i, j, c] >= 0:
                            out[i, j, c, indices_in[i, j, c]] = on_value_const_in
        return out

    if len(shape) == 1:
        out = one_hot_hybrid_1(indices, on_value_const, off_value_const)
    elif len(shape) == 2:
        out = one_hot_hybrid_2(indices, on_value_const, off_value_const)
    elif len(shape) == 3:
        out = one_hot_hybrid_3(indices, on_value_const, off_value_const)
    strategy = onehot_tiling_strategy(out, axis)
    attr_map = {"RewriteVarTensorIdx": True}
    if strategy:
        attr_map["custom_tiling"] = strategy

    return out, attr_map


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                  int, (int, type(None)), (str, type(None)))
def OneHotV2(indices, on_value, off_value, depth, axis=None, target=utils.CCE):
    """
    generate the one-hot code for input indices

    Args:
        indices (akg.tvm.tensor.Tensor): defining the input data.
        on_value (akg.tvm.tensor.Tensor): defining the value to fill in the output if indices[i] == j.
        off_value (akg.tvm.tensor.Tensor): defining the value to fill in the output if indices[i] != j.
        depth (int): defining the depth of the one hot dimension.
        axis (int): optional. The axis to fill. default -1, that means a new inner-most axis.
        attrs (dict): optional.  Dictionary provide tiling information for poly.
        kernel_name (String): optional. the name of the kernel that will be generated.

    Returns:
        akg.tvm.module. A module that combines both host and device code.
    
    Supported Platforms:
        'Ascend'
    """

    utils.ops_dtype_check(indices.dtype, utils.DtypeForDavinci.INT32)
    utils.ops_dtype_check([on_value.dtype, off_value.dtype], [utils.DtypeForDavinci.INT32, utils.DtypeForDavinci.ALL_FLOAT])

    shape = [x.value for x in indices.shape]
    utils.check_shape(shape)

    # OneHot do not support tensor with more than 3 dimensions for now
    if len(shape) > 3:
        raise RuntimeError("one_hot do not support input shape %d dimensions which is more than 3" % len(shape))

    if axis is None:
        axis = -1

    if axis == -1:
        axis = len(shape)

    if axis <= -2 or axis > len(shape):
        raise RuntimeError("axis(%s) is not an valid index" % axis)

    in_shape = [x for x in indices.shape]

    in_shape.insert(axis, depth)
    out_shape = tuple(in_shape)

    @script
    def one_hot_hybrid_1(indices_in, on_value_const_in, off_value_const_in):
        out = output_tensor(out_shape, on_value_const_in.dtype)

        m, n = out_shape

        for i in range(m):
            for j in range(n):
                out[i, j] = off_value_const_in[0]

        if axis == 0:
            for i in range(n):
                if indices_in[i] >= 0:
                    out[indices_in[i], i] = on_value_const_in[0]
        else:
            for i in range(m):
                if indices_in[i] >= 0:
                    out[i, indices_in[i]] = on_value_const_in[0]

        return out

    @script
    def one_hot_hybrid_2(indices_in, on_value_const_in, off_value_const_in):

        out = output_tensor(out_shape, on_value_const_in.dtype)

        m, n, k = out.shape

        for x in range(m):
            for y in range(n):
                for z in range(k):
                    out[x, y, z] = off_value_const_in[0]

        if axis == 0:
            for i in range(n):
                for j in range(k):
                    if indices_in[i, j] >= 0:
                        out[indices_in[i, j], i, j] = on_value_const_in[0]
        elif axis == 1:
            for i in range(m):
                for j in range(k):
                    if indices_in[i, j] >= 0:
                        out[i, indices_in[i, j], j] = on_value_const_in[0]
        else:
            for i in range(m):
                for j in range(n):
                    if indices_in[i, j] >= 0:
                        out[i, j, indices_in[i, j]] = on_value_const_in[0]

        return out

    @script
    def one_hot_hybrid_3(indices_in, on_value_const_in, off_value_const_in):
        out = output_tensor(out_shape, on_value_const_in.dtype)
        m, n, k, t = out.shape

        for x in range(m):
            for y in range(n):
                for z in range(k):
                    for u in range(t):
                        out[x, y, z, u] = off_value_const_in[0]

        if axis == 0:
            for i in range(n):
                for j in range(k):
                    for c in range(t):
                        if indices_in[i, j, c] >= 0:
                            out[indices_in[i, j, c], i, j, c] = on_value_const_in[0]
        elif axis == 1:
            for i in range(m):
                for j in range(k):
                    for c in range(t):
                        if indices_in[i, j, c] >= 0:
                            out[i, indices_in[i, j, c], j, c] = on_value_const_in[0]
        elif axis == 2:
            for i in range(m):
                for j in range(n):
                    for c in range(t):
                        if indices_in[i, j, c] >= 0:
                            out[i, j, indices_in[i, j, c], c] = on_value_const_in[0]
        else:
            for i in range(m):
                for j in range(n):
                    for c in range(k):
                        if indices_in[i, j, c] >= 0:
                            out[i, j, c, indices_in[i, j, c]] = on_value_const_in[0]
        return out

    if len(shape) == 1:
        out = one_hot_hybrid_1(indices, on_value, off_value)
    elif len(shape) == 2:
        out = one_hot_hybrid_2(indices, on_value, off_value)
    elif len(shape) == 3:
        out = one_hot_hybrid_3(indices, on_value, off_value)
    strategy = onehot_tiling_strategy(out, axis)
    attr_map = {"RewriteVarTensorIdx": True}
    if strategy:
        attr_map["custom_tiling"] = strategy

    return out, attr_map
