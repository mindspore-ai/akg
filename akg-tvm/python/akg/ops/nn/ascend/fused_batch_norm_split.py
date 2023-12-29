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

"""operator dsl function: fused_batch_norm_split"""
from __future__ import absolute_import
from functools import reduce

import akg
import akg.tvm
import akg.lang.ascend
import akg.utils as utils
import akg.utils.kernel_exec as kernel_exec
from akg.utils.dsl_create import TensorUtils
from akg.ops.math.rsqrt import rsqrt
from akg.utils import custom_tiling as ct_util
from akg.utils.dsl_create import mul_axis_sum, update_by_moving_average
from akg.utils.format_transform import get_shape

DIM_MAP_BN1 = {
    # dim for 5d
    str(((32, 4, 112, 112, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 4, 1), (0, 4, 112, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 4, 1), (1, 4, 112, 1),
    ),
    str(((32, 4, 56, 56, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 8, 1), (0, 4, 56, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 8, 1), (1, 4, 56, 1),
    ),
    str(((32, 16, 56, 56, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 8, 1), (0, 4, 56, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 8, 1), (1, 4, 56, 1),
    ),
    str(((32, 8, 28, 28, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 28, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 14, 1), (1, 4, 28, 1),
    ),
    str(((32, 32, 28, 28, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 28, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 14, 1), (1, 4, 28, 1),
    ),
    str(((32, 16, 14, 14, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 14, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 14, 1), (1, 4, 14, 1),
    ),
    str(((32, 64, 14, 14, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 14, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 14, 1), (1, 4, 14, 1),
    ),
    str(((32, 32, 7, 7, 16))): (
        (0, 0, 1, 1), (0, 1, 16, 1), (0, 2, 16, 1), (0, 3, 7, 1), (0, 4, 7, 1),
        (1, 0, 1, 1), (1, 1, 16, 1), (1, 2, 16, 1), (1, 3, 7, 1), (1, 4, 7, 1),
    ),
    str(((32, 128, 7, 7, 16))): (
        (0, 0, 1, 1), (0, 1, 16, 1), (0, 2, 16, 1), (0, 3, 7, 1), (0, 4, 7, 1),
        (1, 0, 1, 1), (1, 1, 16, 1), (1, 2, 16, 1), (1, 3, 7, 1), (1, 4, 7, 1),
    ),

    # resnet50 V1.5
    str(((32, 8, 56, 56, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 8, 1), (0, 4, 56, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 8, 1), (1, 4, 56, 1),
    ),
    str(((32, 16, 28, 28, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 28, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 14, 1), (1, 4, 28, 1),
    ),
    str(((32, 32, 14, 14, 16))): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 14, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 16, 1), (1, 3, 14, 1), (1, 4, 14, 1),
    ),
}

ATTR_MAP_BN1 = {
    "float32": {
        "merge_outer_loop_for_multicore": 1,
    }
}

DEFAULT_ATTR_MAP_BN1 = {
    "enable_bisect_optimize": 0,
}


def bn1_tiling_strategy(tensor):
    """Custom tiling strategy for first part of splited fused_batch_norm op"""
    # bn1 input [N, C1, H, W, C0]
    n_pos = 0
    c0_pos = 4
    strategy = list()
    strategy += ct_util.create_constraint_on_tensor(
        tensor=tensor,
        values=1,
        constraints=ct_util.TileConstraint.FACTOR,
        tensor_pos=n_pos)
    strategy += ct_util.create_constraint_on_tensor(
        tensor=tensor,
        values="FULL",
        constraints=ct_util.TileConstraint.MAX,
        tensor_pos=c0_pos)
    return strategy


def bn1_set_dim_func(data):
    """bn1 dim func"""
    hash_key = data.dtype
    if hash_key in ATTR_MAP_BN1.keys():
        attrs_dict = ATTR_MAP_BN1.get(hash_key, {})
        for attr, value in attrs_dict.items():
            DEFAULT_ATTR_MAP_BN1[attr] = value
    hash_key = str((tuple(get_shape(data))))
    if hash_key in DIM_MAP_BN1.keys():
        diminfo = ct_util.set_dims(DIM_MAP_BN1.get(hash_key, {}))
    else:
        diminfo = ""

    return diminfo, hash_key


def bn1_check(data):
    """check bn1 func's parameters availability for fused_bn1"""
    shape = get_shape(data)
    dtype = data.dtype

    if len(shape) != 5:
        raise RuntimeError("Only support 5D data, "
                           "but get {}!".format(shape))

    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)


@utils.check_input_type(akg.tvm.tensor.Tensor, (str, type(None)))
def fused_bn1(data):
    """
    Fused_batch_norm is departed to 3 parts for better performance.

    First part:

    .. math::
        \\begin{array}{ll} \\\\
            m = N \\times H \\times W \\\\
            \\mu_{tmp} = \\sum_{n, h, w}{\\frac{x}{m}} \\\\
            \\sigma^2_{tmp} = \\sum_{n, h, w}{\\frac{x^2}{m}}
        \\end{array}

    Second part:

    .. math::
        \\begin{array}{ll} \\\\
            \\sigma^2 = \\sigma^2_{tmp} - \\mu^2 \\\\
            \\mu_{r} = momentum \\cdot \\mu_{r} + (1-momentum) \\cdot \\mu \\\\
            \\sigma^2_{r} = momentum \\cdot \\sigma^2_{r}
                + (1-momentum) \\cdot \\sigma^2
        \\end{array}

    Third part:

    .. math::
        \\begin{array}{ll} \\\\
            \\hat{\\gamma} =
                \\gamma \\cdot \\frac{1}{\\sqrt{\\sigma^2 + \\epsilon}} \\\\
            \\hat{\\beta} = \\beta - \\hat{\\gamma} \\cdot \\mu \\\\
            res = \\hat{\\gamma} \\cdot x + \\hat{\\beta}
        \\end{array}

    The first part of fused batch norm. It will reduce H and W axis firstly.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16 or float32 with shape
                                  (N,C1,H,W,C0).

    Returns:
        mean (tvm.tensor.Tensor): Tensor of type float32 with shape(1,C1,1,1,C0).
        var_part (tvm.tensor.Tensor): Tensor of type float32 with shape(1,C1,1,1,C0).
    """
    bn1_check(data)
    dim_info, _ = bn1_set_dim_func(data)
    attrs = {**DEFAULT_ATTR_MAP_BN1}
    shape = get_shape(data)

    num = reduce(lambda i, j: i * j, [shape[i] for i in [3, 2, 0]])
    avg_num = float(1) / float(num)

    attrs["custom_tiling"] = bn1_tiling_strategy(data)
    data = data.astype("float32")

    square = akg.tvm.compute(data.shape,
                             lambda *i: data[i] * data[i], name="square")

    axes = [2, 3]
    mean_tmp = mul_axis_sum(data, axes, True)
    var_part_tmp = mul_axis_sum(square, axes, True)
    mean_tmp_div_num = akg.lang.ascend.vmuls(mean_tmp, avg_num)
    var_tmp_div_num = akg.lang.ascend.vmuls(var_part_tmp, avg_num)

    mean = mul_axis_sum(mean_tmp_div_num, [0], True, name="mean")
    var_part = mul_axis_sum(var_tmp_div_num, [0], True, name="var_part")

    if dim_info != "":
        attrs["dim"] = dim_info
    attrs['enable_atomic_add'] = True
    return mean, var_part, attrs


DIM_MAP_BN2 = {
}

ATTR_MAP_BN2 = {
}


def inplace_operate_bind(in_tensors, out_tensors, inplace_binds):
    """
    Some tensor need to be calculate inplace.

    Args:
        in_tensors (Union[list, tuple]): Origin input tensors.
        out_tensors (Union[list, tuple]): Origin output tensors.
        inplace_binds (tuple): Should be a tuple of tuples, the first value
                               of each element is input tensor index, the
                               second is output tensor index,
                               consist (in_id, out_id),
                               meanning out_id output tensor is inplace
                               update to in_id input tensor.
    Returns:
        Two elements tuple, one for output tensors, the other for tensor bind relations.
    """

    for in_id, out_id in inplace_binds:
        if in_id >= len(in_tensors) or out_id >= len(out_tensors):
            raise RuntimeError("Inplace binds is invalid, while there are {} "
                               "input tensors and {} output tensors, but get "
                               "bind {}.".format(
                                   len(in_tensors), len(out_tensors),
                                   inplace_binds))

    out_tensors = list(out_tensors)
    tensor_binds = {}
    inplaced_tensors = []

    for i, bind in enumerate(inplace_binds):
        in_tensor = in_tensors[bind[0]]
        out_tensor = out_tensors[bind[1]]
        out_tensor, binds_info = TensorUtils.inplace_set(
            in_tensor, out_tensor, buffer_name="inp_buf_{}".format(i))
        tensor_binds.update(binds_info)
        # Caculation is updated inplace in input tensor. But Mindspore
        # needs a related fake tensor(never use) in output list...
        out_tensor_shape = out_tensor.shape

        fake_tensor = akg.tvm.compute(
            out_tensor_shape,
            lambda *index, o_tensor=out_tensor: o_tensor(*index),
            name="fake_tensor_{}".format(i))

        out_tensors[bind[1]] = fake_tensor
        inplaced_tensors.append(out_tensor)

    return (tuple(out_tensors + inplaced_tensors), tensor_binds)


def bn2_set_dim_func(*args):
    """bn2 dim func"""
    hash_key = str((tuple(get_shape(args[0]))))
    if hash_key in DIM_MAP_BN2.keys():
        diminfo = ct_util.set_dims(DIM_MAP_BN2.get(hash_key, {}))
    else:
        diminfo = ""

    return diminfo, hash_key


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        (float, type(None)), (str, type(None)))
def fused_bn2(mean, var_part, running_mean, running_var, momentum=0.8):
    """
    Calculating mean, variance and update running variables.

    Read fused_bn1 docs for details.

    Note:
        Apply reduction of 'N' axis to calculating mean and variance.

    Args:
        mean (tvm.tensor.Tensor): Tensor of type float32 as mean.
        var_part (tvm.tensor.Tensor): Tensor of type float32, intermediate
                                      variables for variance.
        running_mean (tvm.tensor.Tensor): Tensor of type float32 as trained
                                          mean used in inference stage.
        running_var (tvm.tensor.Tensor): Tensor of type float32 as trained
                                         variance used in inference stage.
        momentum (float): A float number used for updating running values,
                          must meet condition '0.0 < momentum < 1.0'.

    Returns:
        variance (tvm.tensor.Tensor): A float32 tensor as data's variance.
        running_mean_updated (tvm.tensor.Tensor): A float32 tensor as updated
                                                  running_mean (updated inplace).
        running_var_updated (tvm.tensor.Tensor): A float32 tensor, updated
                                                 running_var (updated inplace).
    """
    utils.ops_dtype_check([mean.dtype, var_part.dtype],
                          utils.DtypeForDavinci.FLOAT32)

    dim_info, _ = bn2_set_dim_func(mean, var_part,
                                   running_mean, running_var, momentum)
    attrs = {**ATTR_MAP_BN2}

    in_tensors = (var_part, mean, running_mean, running_var)

    sub_mean_square = akg.tvm.compute(mean.shape,
                                      lambda *i:
                                      akg.tvm.const(-1.0, dtype=mean.dtype) *
                                      mean(*i) * mean(*i),
                                      name="sub_mean_square")
    variance = akg.tvm.compute(mean.shape,
                               lambda *i: var_part(*i) + sub_mean_square(*i),
                               name="variance")

    # update running mean and variance
    running_mean_updated = \
        update_by_moving_average(running_mean, mean, momentum)
    running_var_updated = \
        update_by_moving_average(running_var, variance, momentum)

    out_tensors = (variance, running_mean_updated, running_var_updated)
    tensors_and_binds = inplace_operate_bind(
        in_tensors, out_tensors, ((2, 1), (3, 2)))
    out_tensors = tensors_and_binds[0]
    attrs[kernel_exec.BINDS] = tensors_and_binds[1]

    if dim_info != "":
        attrs["dim"] = dim_info
    return (*out_tensors, attrs)


DIM_MAP_BN3 = {
    str(((32, 4, 112, 112, 16))): (
        (0, 0, 4, 1), (0, 1, 16, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 112, 1),
    ),

    str(((32, 4, 56, 56, 16))): (
        (0, 0, 4, 1), (0, 1, 16, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 56, 1),
    ),
    str(((32, 16, 56, 56, 16))): (
        (0, 0, 16, 1), (0, 1, 16, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 56, 1),
    ),
    str(((32, 8, 28, 28, 16))): (
        (0, 0, 8, 1), (0, 1, 16, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 28, 1),
    ),
    str(((32, 32, 28, 28, 16))): (
        (0, 0, 1, 1), (0, 1, 16, 1), (0, 2, 32, 1), (0, 3, 1, 1), (0, 4, 28, 1),
    ),
    str(((32, 16, 14, 14, 16))): (
        (0, 0, 16, 1), (0, 1, 16, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 14, 1),
    ),
    str(((32, 64, 14, 14, 16))): (
        (0, 0, 1, 1), (0, 1, 16, 1), (0, 2, 32, 1), (0, 3, 2, 1), (0, 4, 14, 1),
    ),
    str(((32, 32, 7, 7, 16))): (
        (0, 0, 1, 1), (0, 1, 16, 1), (0, 2, 32, 1), (0, 3, 1, 1), (0, 4, 7, 1),
    ),
    str(((32, 128, 7, 7, 16))): (
        (0, 0, 1, 1), (0, 1, 16, 1), (0, 2, 4, 1), (0, 3, 7, 1), (0, 4, 7, 1),
    ),

    # resnet50 V1.5
    str(((32, 8, 56, 56, 16))): (
        (0, 0, 8, 1), (0, 1, 16, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 56, 1),
    ),
    str(((32, 16, 28, 28, 16))): (
        (0, 0, 16, 1), (0, 1, 16, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 28, 1),
    ),
    str(((32, 32, 14, 14, 16))): (
        (0, 0, 1, 1), (0, 1, 16, 1), (0, 2, 32, 1), (0, 3, 2, 1), (0, 4, 14, 1),
    ),
}

ATTR_MAP_BN3 = {
    str(((32, 4, 112, 112, 16))): (
        ("eleminate_outmost_for_cond", True), ("pragma_modshift", 1)),
    str(((32, 4, 56, 56, 16))): (
        ("eleminate_outmost_for_cond", True), ("pragma_modshift", 1)),
    str(((32, 16, 56, 56, 16))): (
        ("eleminate_outmost_for_cond", True), ("pragma_modshift", 1)),
    str(((32, 8, 28, 28, 16))): (
        ("eleminate_outmost_for_cond", True), ("pragma_modshift", 1)),
    str(((32, 16, 14, 14, 16))): (
        ("eleminate_outmost_for_cond", True), ("pragma_modshift", 1)),
    str(((32, 8, 56, 56, 16))): (
        ("eleminate_outmost_for_cond", True), ("pragma_modshift", 1)),
    str(((32, 16, 28, 28, 16))): (
        ("eleminate_outmost_for_cond", True), ("pragma_modshift", 1)),
}

DEFAULT_ATTR_MAP_BN3 = {
}


def bn3_set_dim_func(*args):
    """dim func for fused_bn3"""
    hash_key = str((tuple(get_shape(args[0]))))
    if hash_key in ATTR_MAP_BN3.keys():
        attrs_dict = ATTR_MAP_BN3.get(hash_key, {})
        for attr in attrs_dict:
            DEFAULT_ATTR_MAP_BN3[attr[0]] = attr[1]
    if hash_key in DIM_MAP_BN3.keys():
        dim = ct_util.set_dims(DIM_MAP_BN3.get(hash_key, {}))
    else:
        dim = ""
    return dim, hash_key


def bn3_check(data, mean, variance, gamma, beta):
    """check fused_bn3's parameters availability"""
    shape = get_shape(data)
    dtype = data.dtype

    if len(shape) != 5:
        raise RuntimeError("Only support 5D data, "
                           "but get {}!".format(shape))

    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.ops_dtype_check([variance.dtype, mean.dtype, gamma.dtype, beta.dtype],
                          utils.DtypeForDavinci.FLOAT32)


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        akg.tvm.tensor.Tensor, (float, type(None)), (str, type(None)))
def fused_bn3(data, mean, variance, gamma, beta, eps=1e-3):
    """
    The third part of fused batch norm, calculate the normalized result.

    Read fused_bn1 docs for details.

    Note:
        This part is also the reference implement for fused_batch_norm!

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16 or float32 with
                                  \"NC1HWC0\" format.
        mean (tvm.tensor.Tensor): Tensor of type float32, data's mean.
        variance (tvm.tensor.Tensor): Tensor of type float32, data's variance.
        gamma (tvm.tensor.Tensor): Tensor of type float32 for scaling.
        beta (tvm.tensor.Tensor): Tensor of type float32 for bias.
        eps (float): small float value to avoid dividing zero.

    Returns:
        Tensor as normalized, scaled, shifted data.
    """
    bn3_check(data, mean, variance, gamma, beta)
    dim_info, _ = bn3_set_dim_func(data, mean, variance, gamma, beta, eps)
    attrs = {**DEFAULT_ATTR_MAP_BN3}

    ori_dtype = data.dtype

    # calculate batch norm result
    rsd = rsqrt(akg.tvm.compute(variance.shape,
                                lambda *i:
                                variance(*i) +
                                akg.tvm.const(eps, dtype=variance.dtype),
                                name="var_eps"), utils.CCE)

    hat_gamma = akg.tvm.compute(gamma.shape,
                                lambda *i: gamma(*i) * rsd(*i),
                                name="hat_gamma", attrs={'no_inline': 1})

    hat_beta = akg.tvm.compute(gamma.shape,
                               lambda *i: beta(*i) - hat_gamma(*i) * mean(*i),
                               name="hat_beta", attrs={'no_inline': 1})

    hat_gamma_bc = akg.lang.ascend.broadcast(hat_gamma, data.shape)
    hat_beta_bc = akg.lang.ascend.broadcast(hat_beta, data.shape)

    data_fp32 = akg.tvm.compute(data.shape,
                                lambda *i: data(*i).astype("float32"),
                                name="data_fp32")

    bn_res_fp32 = akg.tvm.compute(data.shape,
                                  lambda *i:
                                  akg.lang.ascend.vmadd(
                                      data_fp32(*i), hat_gamma_bc(*i), hat_beta_bc(*i)),
                                  name="bn_res_fp32")
    res = akg.tvm.compute(bn_res_fp32.shape,
                          lambda *i: bn_res_fp32(*i).astype(ori_dtype),
                          name="bn_res")
    if dim_info != "":
        attrs["dim"] = dim_info
    return res, attrs
