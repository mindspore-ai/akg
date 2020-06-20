#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function: fused_batch_norm_grad"""
from functools import reduce

import akg
import akg.tvm
import akg.topi
from akg.utils import custom_tiling as ct_util, kernel_exec as utils, \
    validation_check as vc_util
from akg.ops.math.add import add
from akg.dim import DIM
from akg.utils.format_transform import get_shape

def check_inputs(dy, data, mean, var, gamma, data_format, axis):
    """check inputs"""
    vc_util.ops_dtype_check([data.dtype, dy.dtype],
                            vc_util.DtypeForDavinci.ALL_FLOAT)
    if data.dtype != dy.dtype:
        raise TypeError(
            "fused_batch_norm_grad require same dtype for dy and data.")
    vc_util.ops_dtype_check([mean.dtype, var.dtype, gamma.dtype],
                            vc_util.DtypeForDavinci.FLOAT32)

    dataformat_checklist = ["NHWC", "NC1HWC0", "NCHW", "DefaultFormat"]
    if data_format not in dataformat_checklist:
        raise RuntimeError(
            "fused_batch_norm_grad only support %s while data_format is %s" %
            (",".join(dataformat_checklist), data_format))

    shape = get_shape(data)
    in_rank = len(shape)
    is_special5d = (data_format == "NC1HWC0")

    if in_rank <= 1:
        raise RuntimeError("do not support 1D data.")
    if data_format == "DefaultFormat":
        if not isinstance(axis, int):
            raise RuntimeError('axis should be instance of int')
        if axis not in range(-in_rank, in_rank):
            raise RuntimeError(
                'axis must be in range [%d, %d)' % (-in_rank, in_rank))
        if axis < 0:
            axis = in_rank + axis
    elif data_format == "NHWC":
        if in_rank != 4:
            raise RuntimeError("data shape {} mismatch data_format \"NHWC\"."
                               "".format(data.shape))
        axis = 3
    elif data_format == "NCHW":
        if in_rank != 4:
            raise RuntimeError("data shape {} mismatch data_format \"NCHW\"."
                               "".format(data.shape))
        axis = 1
    else:
        axis = 1

    if any([x.value != y.value for x, y in zip(data.shape, dy.shape)]):
        raise RuntimeError("the shape of data and dy must be equal.")
    if is_special5d:
        def is_all_1_but_axis_equal(shape1, shape2, axis):
            if not isinstance(axis, (list, tuple)):
                axis = (axis,)
            return all([int(shape1[i]) == 1 if i not in axis else
                        int(shape1[i]) == int(shape2[i]) for i in range(len(shape2))])
        if len(data.shape) != 5:
            raise RuntimeError("data shape {} mismatch data_format \"NC1HWC0\"."
                               "".format(data.shape))
        if len(gamma.shape) != 5 \
                or not is_all_1_but_axis_equal(gamma.shape, shape, (1, 4)):
            raise RuntimeError("gamma mismatch NC1HWC0 data (while gamma shape "
                               "is {}, inputs shape is {})!".format(
                                   gamma.shape, data.shape))
        if len(mean.shape) != 5 \
                or not is_all_1_but_axis_equal(mean.shape, shape, (1, 4)):
            raise RuntimeError("mean mismatch NC1HWC0 data (while mean shape "
                               "is {}, inputs shape is {})!".format(
                                   mean.shape, data.shape))
        if len(var.shape) != 5 \
                or not is_all_1_but_axis_equal(var.shape, shape, (1, 4)):
            raise RuntimeError("var mismatch NC1HWC0 data (while var shape is "
                               "{}, inputs shape is {})!".format(
                                   var.shape, data.shape))
    else:
        if len(gamma.shape) != 1 or (gamma.shape[0].value != shape[axis]):
            raise RuntimeError("gamma mismatch the channel axis(while gamma "
                               "shape is {}, inputs shape is {}, and axis is {}"
                               ")".format(
                                   gamma.shape, data.shape, axis))
        if len(mean.shape) != 1 or (mean.shape[0].value != shape[axis]):
            raise RuntimeError("mean mismatch the channel axis(while mean "
                               "shape is {}, inputs shape is {}, and axis is "
                               "{})".format(
                                   mean.shape, data.shape, axis))
        if len(var.shape) != 1 or (var.shape[0].value != shape[axis]):
            raise RuntimeError("var mismatch the channel axis(while var shape "
                               "is {}, inputs shape is {}, and axis is {})"
                               "".format(
                                   var.shape, data.shape, axis))

    if is_special5d:
        axes = [3, 2, 0]
        mid_shape = [1, shape[1], 1, 1, shape[4]]
    else:
        axes = [i for i in range(in_rank - 1, -1, -1) if i != axis]
        mid_shape = [1] * in_rank
        mid_shape[axis] = shape[axis]

    out_params = {
        "is_special5d": is_special5d,
        "axis": axis,
        "axes": tuple(axes),
        "mid_shape": mid_shape
    }

    return out_params


set_dim_map_ = {
    # key: (rank, dtype, axis, is_special5d)
    str((2, "float16", 1, False)): ((0, 0, 0, 1), (0, 1, 1, 1),
                                    (1, 0, 0, 1), (1, 1, 1, 1)),
    str((2, "float32", 1, False)): ((0, 0, 0, 1), (0, 1, 1, 1),
                                    (1, 0, 0, 1), (1, 1, 1, 1)),
    str((5, "float16", 1, True)): (
        (0, 0, 0, 1), (0, 1, 0, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 1, 1),
        (1, 0, 0, 1), (1, 1, 0, 1), (1, 2, 1, 1), (1, 3, 1, 1), (1, 4, 1, 1)),
    str((5, "float32", 1, True)): (
        (0, 0, 0, 1), (0, 1, 0, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 1, 1),
        (1, 0, 0, 1), (1, 1, 0, 1), (1, 2, 1, 1), (1, 3, 1, 1), (1, 4, 1, 1)),
    str(((32, 4, 112, 112, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (112, 1)),
    str(((32, 4, 56, 56, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (56, 1)),
    str(((32, 16, 56, 56, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (56, 1)),
    str(((32, 8, 28, 28, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (28, 1)),
    str(((32, 32, 28, 28, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (28, 1)),
    str(((32, 16, 14, 14, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (14, 1)),
    str(((32, 64, 14, 14, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (14, 1)),
    str(((32, 32, 7, 7, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (7, 1)),
    str(((32, 128, 7, 7, 16), "float32")): ((4, 1), (16, 1), (1, 1), (1, 1), (7, 1)),
    str((4, "float16", 1, False)): (
        (0, 0, 0, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 1, 1),
        (1, 0, 0, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 1, 1)),
    str((4, "float32", 1, False)): (
        (0, 0, 0, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 1, 1),
        (1, 0, 0, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 1, 1)),
}


def get_attrs():
    """get attrs config"""
    attrs = {
        "pragma_checkcoincident": 0,
        "pragma_reschedule": 1,
        "pragma_modshift": 1,
        "enable_bisect_optimize": 0,
    }
    return attrs

def set_dim_func_(dy, data, mean, var, gamma, eps=1e-3,
                  data_format="DefaultFormat", axis=1):
    """set dim info"""
    shape = get_shape(data)
    params = check_inputs(dy, data, mean, var, gamma, data_format, axis)
    axis = params["axis"]
    is_special5d = params["is_special5d"]
    hash_key1 = str((tuple(shape), data.dtype))
    if hash_key1 in set_dim_map_:
        return ct_util.set_dims_by_key(hash_key1, set_dim_map_), hash_key1
    hash_key = str((len(shape), data.dtype, axis, is_special5d))
    return ct_util.set_dims_by_key(hash_key, set_dim_map_), hash_key


def sum_data(data, axes, keepdims):
    """sum one axis data at a time"""
    for x in axes:
        data = akg.topi.sum(data, axis=x, keepdims=keepdims)
    return data


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor,
                          (float, type(None)), (str, type(None)),
                          (int, list, tuple, type(None)))
def fused_batch_norm_grad(dy, data, mean, var, gamma,
                          eps=1e-3, data_format="DefaultFormat", axis=1):
    r"""
    Gradient for fused_batch_norm.

    .. math::
        \begin{array}{ll} \\
            \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
            \frac {\partial L} {\partial \gamma} =
                \sum_{i=1}^m \\frac{\partial L}{\partial y_i} \cdot \hat{x_i} \\
            \frac {\partial L} {\partial \beta} =
                \sum_{i=1}^m \\frac{\\partial L}{\partial y_i} \\
            \frac {\partial L} {\partial x_i} =
                \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}
                ( \frac{\partial L}{\partial y_i}
                - \frac{1}{m} \cdot \frac{\partial L}{\partial \beta}
                - \frac{\hat{x_i}}{m} \cdot \frac{\partial L}{\partial \gamma})
        \end{array}

    Note:
        When data_format is \"NC1HWC0\", the `mean`, `var` and `gamma` should be
        5D tensors of shape `(1, C1, 1, 1, C0)`, otherwise, they should be
        1D tensors of shape `(C,)`.

    Args:
        dy (tvm.tensor.Tensor): Tensor of type float16, float32
                                (:math:`\\frac{\\partial L}{\\partial y_i}`).
        data (tvm.tensor.Tensor): Tensor of type float16, float32 (:math:`x`).
        mean (tvm.tensor.Tensor): Tensor, the sample mean of `data`
                                  (:math:`\\mu`).
        var (tvm.tensor.Tensor): Tensor, the sample variance of `data`
                                 (:math:`\\sigma`).
        gamma (tvm.tensor.Tensor): Tensor for scaling (:math:`\\gamma`).
        eps (float): A small float added to variance to avoid dividing by zero
                     (:math:`epsilon`).
        data_format (str): Supported format \"DefaultFormat\", \"NCHW\",
                           \"NHWC\" or \"NC1HWC0\".
        axis (Union[int, list, tuple]): An integer to specify the channel axis
                                        when data_format is \"DefaultFormat\".
                                        List or tuple for \"NC1HWC0\".
                                        Must be in the range
                                        [-rank(data), rank(data)).

    Returns:
        dx (tvm.tensor.Tensor): Tensor of the same shape and type as `data`,
                                representing
                                :math:`\frac {\partial L} {\partial x_i}.
        dgamma (tvm.tensor.Tensor): Tensor of the same shape and type as
                                    `gamma`, representing
                                    :math:`\frac{\partial L}{\partial\gamma}`.
        dbeta (tvm.tensor.Tensor): Tensor of the same shape and type as `gamma`,
                                   representing
                                   :math:`\frac{\partial L}{\partial \beta}`.
    """
    attrs = get_attrs()
    dim_info, _ = set_dim_func_(dy, data, mean, var, gamma,
                                eps, data_format, axis)
    attrs[DIM] = dim_info

    params = check_inputs(dy, data, mean, var, gamma, data_format, axis)

    shape = get_shape(data)
    dtype = "float32"
    ori_dtype = data.dtype
    is_special5d = params["is_special5d"]
    axis = params["axis"]
    axes = params["axes"]
    mid_shape = params["mid_shape"]
    keepdims = bool(is_special5d)

    if ori_dtype != dtype:
        dy = akg.topi.cast(dy, dtype)
        data = akg.topi.cast(data, dtype)
        gamma = akg.topi.cast(gamma, dtype)
        mean = akg.topi.cast(mean, dtype)
        var = akg.topi.cast(var, dtype)
    m = reduce(lambda i, j: i * j, [shape[i] for i in axes])
    neg_m_rec = akg.tvm.const((-1.0 / m), dtype=dtype)
    eps = akg.tvm.const(eps, var.dtype)

    mean = akg.lang.cce.broadcast(akg.topi.reshape(mean, mid_shape), shape)
    gamma = akg.lang.cce.broadcast(akg.topi.reshape(gamma, mid_shape), shape)

    var_plus_eps = akg.tvm.compute(
        var.shape, lambda *i: var(*i) + eps, name="var_plus_eps")

    # 1/sqrt(var + eps)
    v = akg.topi.reshape(var_plus_eps, mid_shape)
    if utils.product_is_mini():
        v = akg.topi.cast(v, "float16")
    rsqvar = akg.tvm.compute(mid_shape,
                             lambda *i: akg.tvm.exp(akg.tvm.log(v(*i)) * akg.tvm.const(-0.5, v.dtype)),
                             name="rsqvar",
                             attrs={'no_inline': 1})
    if utils.product_is_mini():
        rsqvar = akg.topi.cast(rsqvar, "float32")

    # data - mean
    data_minus_mean = add(data, mean, -1.0)[0]

    # dbeta = sum(dy)
    dbeta = sum_data(dy, axes, keepdims=keepdims)
    dbeta_bc = akg.lang.cce.broadcast(akg.topi.reshape(dbeta, mid_shape), shape)

    # dgamma = sum(dy * norm)
    dgamma_param = akg.tvm.compute(shape,
                                   lambda *i: dy(*i) * data_minus_mean(*i),
                                   name="dgamma_param")
    dgamma_param_sum = sum_data(dgamma_param, axes, keepdims=True)
    dgamma = akg.tvm.compute(dgamma_param_sum.shape,
                             lambda *i: dgamma_param_sum(*i) * rsqvar(*i),
                             name="dgamma")
    dgamma_bc = akg.lang.cce.broadcast(dgamma, shape)
    rsqvar_bc = akg.lang.cce.broadcast(rsqvar, shape)
    if not is_special5d:
        dgamma = akg.topi.reshape(dgamma, dbeta.shape)

    def cal_dx(*i):
        a = dy(*i)
        b = neg_m_rec * dbeta_bc(*i)
        c = neg_m_rec * rsqvar_bc(*i) * data_minus_mean(*i) * dgamma_bc(*i)
        res = rsqvar_bc(*i) * gamma(*i) * (a + b + c)
        return res

    dx = akg.tvm.compute(shape, cal_dx, name="dx")

    if dtype != ori_dtype:
        dgamma = akg.topi.cast(dgamma, ori_dtype)
        dbeta = akg.topi.cast(dbeta, ori_dtype)
        dx = akg.topi.cast(dx, ori_dtype)
    return dx, dgamma, dbeta, attrs
