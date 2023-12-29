#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""
custom tiling function
"""
from enum import Enum, unique
from functools import wraps
from numpy.core import double
import akg
from akg import dim
from akg.utils.validation_check import check_input_type

set_dim_func_map = {}
gen_key_func_map = {}

NODE_TYPE = "CustomTilingNode"
DEFAULT_VALUE = -1
DEFAULT_STRING = ""
BLOCK_SIZE = 32
CUBE_UNIT = 16


class TileTemplate(Enum):
    """class TileTemplate."""
    NC1HWC0 = "NC1HWC0"
    NCHW = "NCHW"
    DEFAULT_FORMAT = "NCHW"
    NHWC = "NHWC"


@unique
class TileLevel(Enum):
    """class TileLevel."""
    C1 = "C1"
    C0 = "C0"


@unique
class TileMode(Enum):
    """class TileMode."""
    AXIS = "AXIS"
    TENSOR = "TENSOR"
    COMMON = "COMMON"


@unique
class TileConstraint(Enum):
    """class TileConstraint."""
    MIN = "MIN"
    MOD = "MOD"
    MAX = "MAX"
    FACTOR = "FACTOR"
    CANDIDATE = "CANDIDATE"
    FORBID_ISOLATE = "FORBID_ISOLATE"
    SET_PRIORITY = "SET_PRIORITY"
    SET_EXPANSION = "SET_EXPANSION"
    SET_MEM_RATIO = "SET_MEM_RATIO"
    SET_AXIS_INFO = "SET_AXIS_INFO"
    THREAD_MIN = "THREAD_MIN"
    THREAD_MAX = "THREAD_MAX"
    THREAD_MOD = "THREAD_MOD"
    BLOCK_MIN = "BLOCK_MIN"
    BLOCK_MAX = "BLOCK_MAX"
    BLOCK_MOD = "BLOCK_MOD"


def _check_constraint(constraint):
    """Check validity of constraint."""
    if constraint not in TileConstraint:
        raise ValueError("Tile constraint must be chosen from {}".format(TileConstraint))


def _check_tensor_type(tensor):
    """Check the type of tensor."""
    hints = "Tensor should be tvm.tensor.Tensor or a list/tuple of tvm.tensor.Tensor."
    if isinstance(tensor, (list, tuple)):
        for t in tensor:
            if not isinstance(t, akg.tvm.tensor.Tensor):
                raise TypeError(hints)
    elif not isinstance(tensor, akg.tvm.tensor.Tensor):
        raise TypeError(hints)


@check_input_type((double, float, int, list), TileConstraint, TileLevel)
def modify_common_constraints(value, constraint, level=TileLevel.C1):
    """api for dsl to modify some default constraint used in auto tiling."""
    _check_constraint(constraint)
    if constraint == TileConstraint.SET_MEM_RATIO:
        return create_custom_tiling_node(TileMode.COMMON, tile_level=level, mem_ratio=double(value))
    if constraint == TileConstraint.THREAD_MIN:
        return create_custom_tiling_node(TileMode.COMMON, thread_min=value)
    if constraint == TileConstraint.THREAD_MAX:
        return create_custom_tiling_node(TileMode.COMMON, thread_max=value)
    if constraint == TileConstraint.THREAD_MOD:
        return create_custom_tiling_node(TileMode.COMMON, thread_mod=value)
    if constraint == TileConstraint.BLOCK_MIN:
        return create_custom_tiling_node(TileMode.COMMON, block_min=value)
    if constraint == TileConstraint.BLOCK_MAX:
        return create_custom_tiling_node(TileMode.COMMON, block_max=value)
    if constraint == TileConstraint.BLOCK_MOD:
        return create_custom_tiling_node(TileMode.COMMON, block_mod=value)
    raise TypeError("Constraint {} is not supported in this api, please use other api".format(constraint.value))


@check_input_type((str, int), TileConstraint, int, (int, list, tuple, type(None)), TileLevel)
def create_constraint_on_axis(values, constraints, band=0, axis=None, level=TileLevel.C1):
    """api for dsl to create tiling constraints on certain axis."""
    _check_constraint(constraints)

    res = []
    if axis is None:
        axis = [i for i in range(len(values))]
    elif not isinstance(axis, (int, list, tuple)):
        raise TypeError("Axis should be int, list or tuple")

    if isinstance(axis, int):
        axis = [axis]

    if isinstance(values, (str, int)):
        values = [values]
    else:
        raise TypeError("Tiling factor must be string or int, while receives {}".format(type(values)))

    if len(axis) != len(values):
        raise ValueError("Length of axis must equal to length of values")

    for a, v in zip(axis, values):
        if constraints == TileConstraint.MIN:
            res.append(create_custom_tiling_node(TileMode.AXIS, tile_level=level,
                                                 tile_band=band, tile_axis=a, tile_min=v))
        elif constraints == TileConstraint.MOD:
            res.append(create_custom_tiling_node(TileMode.AXIS, tile_level=level,
                                                 tile_band=band, tile_axis=a, tile_mod=v))
        elif constraints == TileConstraint.FACTOR:
            res.append(create_custom_tiling_node(TileMode.AXIS, tile_level=level,
                                                 tile_band=band, tile_axis=a, tile_factor=v))
        elif constraints == TileConstraint.CANDIDATE:
            res.append(create_custom_tiling_node(TileMode.AXIS, tile_level=level,
                                                 tile_band=band, tile_axis=a, tile_candidate=v))
        elif constraints == TileConstraint.MAX:
            res.append(create_custom_tiling_node(TileMode.AXIS, tile_level=level,
                                                 tile_band=band, tile_axis=a, tile_max=v))
        elif constraints == TileConstraint.FORBID_ISOLATE:
            res.append(create_custom_tiling_node(TileMode.AXIS, tile_level=level,
                                                 tile_band=band, tile_axis=a, forbid_isolate=v))
        elif constraints == TileConstraint.SET_AXIS_INFO:
            res.append(create_custom_tiling_node(TileMode.AXIS, tile_level=level,
                                                 tile_band=band, tile_axis=a, axis_info=v))
        elif constraints == TileConstraint.SET_PRIORITY:
            res.append(create_custom_tiling_node(TileMode.AXIS, tile_level=level,
                                                 tile_band=band, tile_axis=a, priority=v))
        else:
            raise TypeError("Constraint {} is not supported in this api, please use other api"
                            .format(constraints.value))
    return res


@check_input_type((akg.tvm.tensor.Tensor, list, tuple), (str, int, list, tuple), TileConstraint,
                  (int, list, tuple, type(None)), TileLevel)
def create_constraint_on_tensor(tensor, values, constraints, tensor_pos=None, level=TileLevel.C1):
    """api for dsl to create tiling constraints on certain tensor."""
    _check_constraint(constraints)
    _check_tensor_type(tensor)

    tensor_name = [tensor.op.name] if isinstance(tensor, akg.tvm.tensor.Tensor) else [t.op.name for t in tensor]
    values = [values] if isinstance(values, (str, int)) else values

    if tensor_pos is None:
        tensor_pos = [i for i in range(len(values))]
    else:
        tensor_pos = [tensor_pos] if isinstance(tensor_pos, int) else tensor_pos
        if len(tensor_pos) != len(values):
            raise ValueError("Length of tensor position is not compatible with length of constraint values")

    strategy = list()
    for t in tensor_name:
        for p, v in zip(tensor_pos, values):
            if constraints == TileConstraint.MIN:
                strategy.append(create_custom_tiling_node(TileMode.TENSOR, tile_level=level,
                                                          tensor_name=t, tile_pos=p, tile_min=v))
            elif constraints == TileConstraint.MOD:
                strategy.append(create_custom_tiling_node(TileMode.TENSOR, tile_level=level,
                                                          tensor_name=t, tile_pos=p, tile_mod=v))
            elif constraints == TileConstraint.FACTOR:
                strategy.append(create_custom_tiling_node(TileMode.TENSOR, tile_level=level,
                                                          tensor_name=t, tile_pos=p, tile_factor=v))
            elif constraints == TileConstraint.CANDIDATE:
                strategy.append(create_custom_tiling_node(TileMode.TENSOR, tile_level=level,
                                                          tensor_name=t, tile_pos=p, tile_candidate=v))
            elif constraints == TileConstraint.MAX:
                strategy.append(create_custom_tiling_node(TileMode.TENSOR, tile_level=level,
                                                          tensor_name=t, tile_pos=p, tile_max=v))
            elif constraints == TileConstraint.FORBID_ISOLATE:
                strategy.append(create_custom_tiling_node(TileMode.TENSOR, tile_level=level,
                                                          tensor_name=t, tile_pos=p, forbid_isolate=v))
            elif constraints == TileConstraint.SET_PRIORITY:
                strategy.append(create_custom_tiling_node(TileMode.TENSOR, tile_level=level,
                                                          tensor_name=t, tile_pos=p, priority=v))
            elif constraints == TileConstraint.SET_EXPANSION:
                strategy.append(create_custom_tiling_node(TileMode.TENSOR, tile_level=level,
                                                          tensor_name=t, expansion=v))
            else:
                raise TypeError("Constraint {} is not supported in this api, please use other api"
                                .format(constraints.value))
    return strategy


@check_input_type(akg.tvm.tensor.Tensor, TileTemplate, TileLevel)
def create_template(tensor, template, level=TileLevel.C1):
    """create template according to given template arg."""
    tensor_name = tensor.op.name
    if template not in TileTemplate:
        raise ValueError("Invalid template name {0}, must chosen from {1}".
                         format(template, TileTemplate))
    if template in [TileTemplate.NCHW, TileTemplate.DEFAULT_FORMAT]:
        return template_nchw(tensor_name, level)
    if template == TileTemplate.NC1HWC0:
        return template_nc1hwc0(tensor_name, level)
    if template == TileTemplate.NHWC:
        return template_nhwc(tensor_name, level)
    return []


def to_tvm_type(value, t_type):
    """transform integer and string to corresponding type in tvm."""
    if isinstance(value, int):
        return akg.tvm.expr.IntImm("int32", value)
    if isinstance(value, str):
        return akg.tvm.expr.StringImm(value)
    if isinstance(value, (akg.tvm.expr.IntImm, akg.tvm.expr.StringImm)):
        return value
    raise TypeError("{} only support integer or string, found {}".format(t_type, type(value)))


def create_custom_tiling_node(tile_mode,
                              tile_level=TileLevel.C1,
                              tensor_name=DEFAULT_STRING,
                              tile_pos=DEFAULT_VALUE,
                              tile_band=DEFAULT_VALUE,
                              tile_axis=DEFAULT_VALUE,
                              tile_min=DEFAULT_VALUE,
                              tile_max=DEFAULT_VALUE,
                              tile_mod=DEFAULT_VALUE,
                              tile_factor=DEFAULT_VALUE,
                              tile_candidate=DEFAULT_VALUE,
                              forbid_isolate=DEFAULT_VALUE,
                              axis_info=DEFAULT_STRING,
                              priority=DEFAULT_VALUE,
                              expansion=DEFAULT_VALUE,
                              mem_ratio=double(DEFAULT_VALUE),
                              thread_min=[],
                              thread_max=[],
                              thread_mod=[],
                              block_min=[],
                              block_max=[],
                              block_mod=[]):
    """default method to create custom tiling node, all values are default except tile mode."""

    tile_min = to_tvm_type(tile_min, "tile_min")
    tile_max = to_tvm_type(tile_max, "tile_max")
    tile_mod = to_tvm_type(tile_mod, "tile_mod")
    tile_factor = to_tvm_type(tile_factor, "tile_factor")
    tile_candidate = to_tvm_type(tile_candidate, "tile_candidate")
    return akg.tvm.make.node(NODE_TYPE,
                             tile_level=akg.tvm.expr.StringImm(tile_level.value),
                             tile_mode=akg.tvm.expr.StringImm(tile_mode.value),
                             tensor_name=akg.tvm.expr.StringImm(tensor_name),
                             tile_pos=tile_pos,
                             tile_band=tile_band,
                             tile_axis=tile_axis,
                             tile_min=tile_min,
                             tile_max=tile_max,
                             tile_mod=tile_mod,
                             tile_factor=tile_factor,
                             tile_candidate=tile_candidate,
                             forbid_isolate=forbid_isolate,
                             axis_info=akg.tvm.expr.StringImm(axis_info),
                             priority=priority,
                             expansion=expansion,
                             mem_ratio=mem_ratio,
                             thread_min=thread_min,
                             thread_max=thread_max,
                             thread_mod=thread_mod,
                             block_min=block_min,
                             block_max=block_max,
                             block_mod=block_mod)


def template_nc1hwc0(tensor_name, level):
    """create default tiling strategy for nc1hwc0 template."""
    node_n = create_custom_tiling_node(TileMode.TENSOR,
                                       tile_level=level,
                                       tensor_name=tensor_name,
                                       tile_pos=0,
                                       tile_factor=to_tvm_type(1, "tile_factor"))
    node_c0 = create_custom_tiling_node(TileMode.TENSOR,
                                        tile_level=level,
                                        tensor_name=tensor_name,
                                        tile_pos=4,
                                        tile_max="FULL")
    return [node_n, node_c0]


def template_nchw(tensor_name, level):
    """create default tiling strategy for nchw template."""
    node_n = create_custom_tiling_node(TileMode.TENSOR,
                                       tile_level=level,
                                       tensor_name=tensor_name,
                                       tile_pos=0,
                                       tile_factor=to_tvm_type(1, "tile_factor"))
    node_c = create_custom_tiling_node(TileMode.TENSOR,
                                       tile_level=level,
                                       tensor_name=tensor_name,
                                       tile_pos=1,
                                       tile_mod=to_tvm_type(CUBE_UNIT, "tile_factor"))
    return [node_n, node_c]


def template_nhwc(tensor_name, level):
    """create default tiling strategy for nhwc template."""
    node_n = create_custom_tiling_node(TileMode.TENSOR,
                                       tile_level=level,
                                       tensor_name=tensor_name,
                                       tile_pos=0,
                                       tile_factor=to_tvm_type(1, "tile_factor"))
    node_c = create_custom_tiling_node(TileMode.TENSOR,
                                       tile_level=level,
                                       tensor_name=tensor_name,
                                       tile_pos=3,
                                       tile_mod=to_tvm_type(CUBE_UNIT, "tile_factor"))
    return [node_n, node_c]


def set_dims(tiling):
    """Set dim for tiling."""
    info = dim.Dim()
    for d, tile_d in enumerate(tiling):
        if len(tile_d) == 2:  # only c1 and c0 tile
            index = 0
            axis = d
            c1 = tile_d[0]
            c0 = tile_d[1]
        elif len(tile_d) == 4:  # index, axis, c1, c0
            index = tile_d[0]
            axis = tile_d[1]
            c1 = tile_d[2]
            c0 = tile_d[3]
        else:
            raise RuntimeError("Each element in tiling should be length-2 (c1_tile, c0_tile) "
                               "or length-4 (band_index, axis_index, c1_tile, c0_tile)")
        info.setdim(index=index, axis=axis, tilel1=c1, tilel0=c0)
    return str(info)


def set_dims_by_key(key, map_):
    """Set dim for tiling by key."""
    if key in map_.keys():
        return set_dims(map_[key])
    return ""


def reg_set_dim_func(set_dim_func):
    """register setdim function."""

    def decorate(func_):
        @wraps(func_)
        def wrapper(*args, **kwargs):
            set_dim_func_map[func_.__name__] = set_dim_func
            return func_(*args, **kwargs)

        return wrapper

    return decorate


def reg_set_dim_func_by_func(func_, set_dim_func):
    """register setdim function by function."""
    set_dim_func_map[func_.__name__] = set_dim_func


def reg_gen_key_func(gen_key_func):
    """register generated key by function."""

    def decorate(func_):
        @wraps(func_)
        def wrapper(*args, **kwargs):
            gen_key_func_map[func_.__name__] = gen_key_func
            return func_(*args, **kwargs)

        return wrapper

    return decorate
