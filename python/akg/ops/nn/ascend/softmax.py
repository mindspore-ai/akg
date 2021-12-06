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

"""operator dsl function: softmax"""
import akg.topi
import akg.tvm
import akg
from akg.utils.kernel_exec import product_is_mini
from akg.utils import format_transform as ft_util
import akg.utils as utils
from akg.utils import dynamic_shape as ds
from akg.utils import custom_tiling as ct


def softmax_build(shape, dtype, axis):
    """build softmax."""
    data = akg.tvm.placeholder(shape, name="data", dtype=dtype)

    # compute
    out = softmax(data, axis)
    return out


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple, int), (str, type(None)))
def Softmax(data, axis, target=utils.CCE):
    """
    Map all element of data to (0,1) and sum to 1.

    Args:
        data (tvm.tensor.Tensor): input.
        axis (int): along which normalization is applied.

    Return:
        tvm.tensor.Tensor, output.
    
    Supported Platforms:
        'Ascend'
    """
    utils.check_shape(data.shape)
    shape = data.shape

    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.reduce_axis_check(shape, axis)
    axis = ft_util.refine_reduce_axis(data, axis)

    if isinstance(axis, (list, tuple)):
        if len(axis) != 1:
            raise RuntimeError("Reduce axis for softmax op must be 1-dimension, while current is %d-dimension"
                               % (len(axis)))
        axis = axis[0]
    output = softmax_op(data, axis, shape)
    attr_map = {}
    if ds.shape_is_dynamic(data):
        # For shifted loops, should have:
        #     dynamic_shape_bound mod tile_size_prime == 2
        # This aims to ensure that the shift constant is a multiple of tile_size_prime.
        # So the generated IR will not have complicated head and tail for shifted blocks.
        attr_map = {
            "pragma_modshift": 1,
            "pragma_outerband_need_split": 1,
            "enable_post_poly_loop_partition": False,
            "pragma_disable_whole_component": False,
            "dynamic_shape": ds.set_dynamic_shape_limit_for_tensor(
                output, 2048, axis) +
                             ds.set_poly_upper_bound_for_tensor(
                                 output, 2048, axis),
            "custom_tiling": ct.create_constraint_on_tensor(
                tensor=output,
                values=[
                    1 for i,
                    _ in enumerate(shape) if i != axis],
                constraints=ct.TileConstraint.FACTOR,
                tensor_pos=[
                    i for i,
                    _ in enumerate(shape) if i != axis])}
    return output, attr_map


def softmax_op(data, axis, shape):
    """core computation of softmax op."""
    max_data = akg.lang.ascend.reduce_max(data, axis=axis, keepdims=True)
    max_broadcast = akg.lang.ascend.broadcast(max_data, shape)
    data_sub = akg.lang.ascend.vsub(data, max_broadcast)
    if data.dtype == "float32" and product_is_mini():
        data16 = akg.topi.cast(data_sub, "float16")
        data_exp = akg.lang.ascend.vexp(data16)
        data_exp = akg.topi.cast(data_exp, "float32")
    else:
        data_exp = akg.lang.ascend.vexp(data_sub)

    data_expsum = akg.lang.ascend.sum(data_exp, axis, keepdims=True)
    data_expsum_broadcast = akg.lang.ascend.broadcast(data_expsum, shape)
    output = data_exp / data_expsum_broadcast
    return output
