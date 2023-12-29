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

"""operator dsl function:argmin_argmax_common"""
import akg.tvm
import akg.topi
from akg.lang import ascend as dav
from akg.utils import custom_tiling as ct_util, validation_check as utils
from akg.utils.dsl_create import get_reduce_out_shape
from akg.utils.format_transform import refine_reduce_axis, get_shape
from akg.utils.dynamic_shape import shape_is_dynamic, set_dynamic_shape_limit_for_tensor


def argminmax_tiling_strategy(out_shape, axis):
    """Custom tiling strategy for argminmax op."""
    strategy = list()
    # when reduce axis is one, we do not need any strategy
    if out_shape[axis] == 1:
        return strategy

    # if reduce first axis, it will transpose to last axis
    # so here we adapt to this change
    if axis == 0:
        temp = out_shape[0]
        out_shape = out_shape[1:]
        out_shape.append(temp)
        axis = len(out_shape) - 1

    # eliminate single axis, which will automatically disappear in halide ir
    # and adjust axis if it is influenced
    shrink = list()
    for i, shp in enumerate(out_shape):
        if shp == 1:
            if i < axis:
                axis -= 1
        else:
            shrink.append(shp)

    for i, _ in enumerate(shrink):
        if i == axis:
            strategy.append(ct_util.create_constraint_on_axis(
                values="FULL",
                constraints=ct_util.TileConstraint.MAX,
                axis=i)[0])
        else:
            strategy.append(ct_util.create_constraint_on_axis(
                values=1,
                constraints=ct_util.TileConstraint.FACTOR,
                axis=i)[0])
    return strategy


@utils.check_input_type(akg.tvm.tensor.Tensor, int, (str, type(None)))
def common(data, axis, method="min"):
    """
    Returns the index with the max or min value across axes of a tensor.

    Note:
        method can be "max" or "min" to get argmax or argmin.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32, int8, int32.
        axis (int): Describe the axis of input tensor.
        method (str): Can be "max" or "min".

    Returns:
        tvm.tensor.Tensor, has type of int32.
    """
    shape = get_shape(data)
    dtype = data.dtype

    utils.ops_dtype_check(data.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.ALL_INT])
    utils.reduce_axis_check(shape, axis)
    real_axis = refine_reduce_axis(shape, axis)[0]
    out_shape = get_reduce_out_shape(shape, axis=axis)
    attr_map = {}
    if shape_is_dynamic(data):
        attr_map["dynamic_shape"] = set_dynamic_shape_limit_for_tensor(data, 4096, real_axis)
    if dtype != "float16":
        data = akg.topi.cast(data, "float16")
    k = akg.tvm.reduce_axis((0, data.shape[real_axis]), "k")
    if axis in (len(shape) - 1, -1):
        if method == "min":
            reducer = akg.tvm.comm_reducer(
                lambda x, y: dav.fargmin(x, y), lambda t: akg.tvm.max_value(t))
        elif method == "max":
            reducer = akg.tvm.comm_reducer(
                lambda x, y: dav.fargmax(x, y), lambda t: akg.tvm.min_value(t))
        else:
            raise ValueError("not support {}".format(method))

        if len(data.shape) == 1:
            res = akg.tvm.compute((1,), lambda i: reducer(data[k], axis=k))
        else:
            res = akg.tvm.compute(out_shape,
                                  lambda *indice:
                                  reducer(data(*indice, k), axis=k))

        res = akg.tvm.compute(out_shape,
                              lambda *indice: res(*indice).astype("int32"),
                              "argred_output")
    elif axis in (0, -len(shape)):
        tmp_idx = akg.tvm.compute(shape[1:],
                                  lambda *indice: akg.tvm.const(0.0, "float16"),
                                  name='tmp_index')
        local_data = akg.tvm.compute(shape[1:],
                                     lambda *indice: data(0, *indice),
                                     name="tmp_data")
        for idx in range(shape[axis] - 1):
            if method == 'min':
                tmp_idx = akg.tvm.compute(
                    shape[1:],
                    lambda *indice, ite_idx=idx:
                    akg.tvm.expr.Select(
                        local_data(*indice) > data(ite_idx + 1, *indice),
                        akg.tvm.const(ite_idx + 1, "float16"),
                        tmp_idx(*indice)
                    ))
                local_data = akg.tvm.compute(
                    shape[1:],
                    lambda *indice, ite_idx=idx:
                    akg.tvm.expr.Select(
                        local_data(*indice) > data(ite_idx + 1, *indice),
                        data(ite_idx + 1, *indice),
                        local_data(*indice)
                    ))
            elif method == "max":
                tmp_idx = akg.tvm.compute(
                    shape[1:],
                    lambda *indice, ite_idx=idx:
                    akg.tvm.expr.Select(
                        local_data(*indice) < data(ite_idx + 1, *indice),
                        akg.tvm.const(ite_idx + 1, "float16"),
                        tmp_idx(*indice)
                    ))
                local_data = akg.tvm.compute(
                    shape[1:],
                    lambda *indice, ite_idx=idx:
                    akg.tvm.expr.Select(
                        local_data(*indice) < data(ite_idx + 1, *indice),
                        data(ite_idx + 1, *indice),
                        local_data(*indice)
                    ))
            else:
                raise ValueError("not support " + method)

        res = akg.tvm.compute(out_shape,
                              lambda *indice: tmp_idx(*indice).astype("int32"),
                              "cast1")
    else:
        raise ValueError("Argmax only support first axis and is last axis now!")

    lager = out_shape if len(out_shape) > len(shape) else shape
    strategy = argminmax_tiling_strategy(lager, real_axis)
    if strategy:
        attr_map["custom_tiling"] = strategy
    return res, attr_map
