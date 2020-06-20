#!/usr/bin/env python3
# coding: utf-8
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

"""dynamic shape function"""
import akg
import akg.tvm
from akg.utils.format_transform import get_shape

NODE_TYPE = "DynamicShapeNode"


def to_expanded_list(data):
    data_list = []
    if isinstance(data, (list, tuple)):
        for i in data:
            tmp_list = to_expanded_list(i)
            for ii in tmp_list:
                data_list.append(ii)
    else:
        data_list.append(data)
    return data_list


def shape_is_dynamic(data):
    data_list = to_expanded_list(data)
    for i in data_list:
        shape = get_shape(i)
        if False in [isinstance(s, (int, akg.tvm.expr.IntImm)) for s in shape]:
            return True
    return False


def preprocess_position(position):
    """check position's value is valid and turn integer position into list"""
    if isinstance(position, (list, tuple)):
        for p in position:
            if not isinstance(p, int):
                raise TypeError("Position of tensor should be a integer")
    elif isinstance(position, int):
        position = [position]
    else:
        raise TypeError(
            "Position of tensor should be a integer, list or a tuple")
    return position


def preprocess_value_with_position(values, position):
    """check value is valid and compatible with position, and turn integer into list"""

    if isinstance(values, (list, tuple)):
        if len(values) != len(position):
            raise ValueError(
                "Length of values is not compatible with position.")
        for l in values:
            if not isinstance(l, int):
                raise TypeError(
                    "Dynamic shape values of tensor should be a integer or a list/tuple of integer")
    elif isinstance(values, int):
        values = [values]
    else:
        raise TypeError(
            "Dynamic shape values of tensor should be a integer or a list/tuple of integer")
    return values


def set_poly_upper_bound_for_tensor(tensor, upper_bound, position=None):
    """api for dsl to set poly upper bound for certain tensor."""
    if not isinstance(tensor, akg.tvm.tensor.Tensor):
        raise TypeError("Tensor should be tvm.tensor.Tensor")
    if position is None:
        position = [i for i, _ in enumerate(tensor.shape)]
    position = preprocess_position(position)

    upper_bound = preprocess_value_with_position(upper_bound, position)

    tensor_shape = get_shape(tensor)
    ret = list()
    for i, p in enumerate(position):
        # create limit for var will help poly to determine the upper bound
        if isinstance(tensor_shape[p], akg.tvm.expr.Var):
            ret.append(create_dynamic_shape_node(
                tensor_name=tensor_shape[p].name, pos=p, poly_upper_bound=upper_bound[i]))
    return ret


def set_dynamic_shape_limit_for_tensor(tensor, limit, position=None):
    """api for dsl to set dynamic shape limit for certain tensor."""
    if not isinstance(tensor, akg.tvm.tensor.Tensor):
        raise TypeError("Tensor should be tvm.tensor.Tensor")

    if position is None:
        position = [i for i, _ in enumerate(tensor.shape)]
    position = preprocess_position(position)

    limit = preprocess_value_with_position(limit, position)

    tensor_name = tensor.op.name
    ret = list()
    for i, p in enumerate(position):
        # create limit for tensor in position p will help inferbound to determine the max bound
        ret.append(create_dynamic_shape_node(
            tensor_name=tensor_name, pos=p, dyn_shape_limit=limit[i]))
    return ret


def create_dynamic_shape_node(tensor_name, pos, dyn_shape_limit=-1, poly_upper_bound=-1):
    return akg.tvm.make.node(NODE_TYPE,
                             tensor_name=tensor_name,
                             pos=pos,
                             dyn_shape_limit=dyn_shape_limit,
                             poly_upper_bound=poly_upper_bound)
