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

"""operator dsl function: gather_v2"""
import akg.tvm
import akg.utils as  utils
from akg.utils.format_transform import get_shape
from akg.utils import custom_tiling as ct_util


attrs = {
    "RewriteVarTensorIdx": True,
    "enable_double_buffer": False,
}

gather_v2_set_dim_map = {
}

def gather_v2_set_dim_func(params, indices, axis):
    """set dim info for attr"""
    key = []
    key.append(tuple(params.shape))
    key.append(tuple(indices.shape))
    key.append(axis)
    key.append(params.dtype)
    key.append(indices.dtype)
    hash_key = str(tuple(key))

    if hash_key in gather_v2_set_dim_map.keys():
        return ct_util.set_dims(gather_v2_set_dim_map[hash_key]), hash_key
    return "", hash_key

def gather_tiling_strategy(data, axis):
    """Custom tiling strategy for gather op"""
    strategy = list()
    base = 0
    for priority_value, pos in enumerate(range(len(data.shape) - 1, axis, -1)):
        priority_value = priority_value + base
        strategy.append(ct_util.create_constraint_on_tensor(tensor=data,
                                                            values=priority_value,
                                                            constraints=ct_util.TileConstraint.SET_PRIORITY,
                                                            tensor_pos=pos)[0])
    return strategy


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (int, type(None)), (str, type(None)))
def GatherV2(params, indices, axis=0, target=utils.CCE):
    """
    Select tensor of related dimensions.

    Note:
       Each entry in indices must be an index in [0, params.shape[axis]).

    Args:
       params (tvm.tensor.Tensor): Data to be gathered. Types: int8, int32, float16, float32.
       indices (tvm.tensor.Tensor): A 1-D tensor for index. Types: int32.
                                    Each entry in indices must be an index in [0, params.shape[axis]).
       axis (int): Axis along which index of params be applied. Default: 0.

    Returns:
        tvm.tensor.Tensor, which indexes the input tensor along dimension dim (axis) using the entries in index.
    Supported Platforms:
        'Ascend'
    """
    input_shape = get_shape(params)
    indices_shape = get_shape(indices)
    utils.check_shape(params.shape, tensor_name="params")
    utils.check_shape(indices.shape, length=1, tensor_name="indices")
    utils.ops_dtype_check(params.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.ALL_INT])
    utils.ops_dtype_check(indices.dtype, utils.DtypeForDavinci.INT32)
    axis_num = len(input_shape)
    utils.check_value_on_integer("axis", axis, -axis_num, axis_num)
    if axis < 0:
        axis = axis_num + axis

    def _get_output_shape():
        out_shape = []
        for i, in_shape in enumerate(input_shape):
            if i != axis:
                out_shape.append(in_shape)
            else:
                for indice_shape in indices_shape:
                    out_shape.append(indice_shape)
        return out_shape

    def _get_input_index(output_index):
        input_index = []
        indices_len = len(indices_shape)
        axis_input_index = indices[output_index[axis:axis + indices_len]]
        for i in range(axis):
            input_index.append(output_index[i])
        input_index.append(axis_input_index)
        for i in range(axis + 1, len(output_index)):
            input_index.append(output_index[i])
        return input_index

    output_shape = _get_output_shape()
    output = akg.tvm.compute(
        output_shape, lambda *indices_output: params(*_get_input_index(
            indices_output)), name="gather_output")

    dim_info = gather_v2_set_dim_func(params, indices, axis)[0]
    if dim_info != "":
        attrs['dim'] = dim_info

    attrs["custom_tiling"] = gather_tiling_strategy(params, axis)
    attrs["enable_feature_library"] = True

    return output, attrs
