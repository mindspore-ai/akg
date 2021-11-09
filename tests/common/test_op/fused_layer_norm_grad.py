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

"""operator dsl function: fused_layer_norm_grad"""
from functools import reduce

import akg
import akg.topi
import akg.tvm
from akg import lang
from akg.utils import kernel_exec as utils, custom_tiling as ct_util, \
                      validation_check as vc_util
from akg.dim import DIM
from akg.utils.format_transform import get_shape

fused_layer_norm_grad_set_dim_map = {
    str(([8, 128, 768], 2, 2, "float32")): (
        (0, 0, 8, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 8, 1)),
    str(([8, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([8192, 1024], 1, 1, "float16")): ((0, 0, 1, 1), (0, 1, 2047, 1)),
    str(([2, 128, 768], 2, 2, "float32")): (
        (0, 0, 2, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 2, 1), (1, 2, 1, 1),
        (2, 0, 1535, 1), (2, 1, 1, 1), (2, 2, 2, 1)),
    str(([2, 128, 1024], 2, 2, "float32")): (
        (0, 0, 2, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 2, 1), (1, 2, 1, 1),
        (2, 0, 2047, 1), (2, 1, 1, 1), (2, 2, 2, 1)),
    str(([16, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([16, 128, 768], 2, 2, "float32")): (
        (0, 0, 8, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 8, 1)),
    str(([32, 128, 768], 2, 2, "float32")): (
        (0, 0, 8, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 8, 1)),
    str(([1024, 128, 768], 2, 2, "float32")): (
        (0, 0, 8, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 8, 1)),
    str(([256, 128, 768], 2, 2, "float32")): (
        (0, 0, 8, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 8, 1)),
    str(([512, 128, 768], 2, 2, "float32")): (
        (0, 0, 8, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 8, 1)),
    str(([1024, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([128, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([128, 128, 768], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([256, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([32, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([4, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([4, 128, 768], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([64, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([64, 128, 768], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 1535, 1),
        (1, 0, 1535, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
    str(([512, 128, 1024], 2, 2, "float32")): (
        (0, 0, 4, 1), (0, 1, 1, 1), (0, 2, 2047, 1),
        (1, 0, 2047, 1), (1, 1, 1, 1), (1, 2, 4, 1)),
}

fused_layer_norm_grad_set_attr_map = {
    str(([2, 128], 1, 1, "float32")): (["pragma_disable_whole_component", "disable_cse"]),
    str(([2, 128, 768], 2, 2, "float32")): (["pragma_disable_whole_component", "disable_cse"]),
    str(([2, 128, 1024], 2, 2, "float32")): (["pragma_disable_whole_component", "disable_cse"]),
    str(([8192, 1024], 1, 1, "float16")): (["pragma_disable_whole_component"]),
}

prob_shape = {
    0: [16, 128, 1024],
    1: [16, 128, 768],
    2: [8, 128, 1024],
    3: [32, 128, 1024],
    4: [32, 128, 768],
    5: [128, 128, 1024],
    6: [128, 128, 768],
    7: [256, 128, 1024],
    8: [256, 128, 768],
    9: [512, 128, 1024],
    10: [512, 128, 768],
    11: [1024, 128, 1024],
    12: [1024, 128, 768],
    13: [64, 128, 1024],
    14: [64, 128, 768],
}


def fused_layer_norm_grad_set_dim_func(x, _dy, _variance, _mean, _gamma,
                                       begin_norm_axis, begin_params_axis):
    """dim function"""
    shape = get_shape(x)
    if begin_norm_axis < 0:
        begin_norm_axis = begin_norm_axis + len(shape)
    if begin_params_axis < 0:
        begin_params_axis = begin_params_axis + len(shape)
    hash_key = str((shape, begin_norm_axis, begin_params_axis, x.dtype))
    attr_map = dict()
    attr_map["pragma_checkcoincident"] = 0
    if hash_key in fused_layer_norm_grad_set_attr_map.keys():
        for attr in fused_layer_norm_grad_set_attr_map[hash_key]:
            attr_map[attr] = 1

    return ct_util.set_dims_by_key(hash_key, fused_layer_norm_grad_set_dim_map), hash_key, attr_map


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, int, int)
def fused_layer_norm_grad(x, dy, variance, mean, gamma, begin_norm_axis, begin_params_axis):
    """Gradient of fused layer norm."""
    dim_info, _, attrs = fused_layer_norm_grad_set_dim_func(x, dy, variance, mean, gamma,
                                                            begin_norm_axis, begin_params_axis)
    attrs[DIM] = dim_info

    # check shapes
    shape = get_shape(x)
    vc_util.check_shape(shape)

    in_rank = len(shape)
    # check begin_norm_axis and begin_params_axis
    if abs(begin_norm_axis) >= in_rank or abs(begin_params_axis) >= in_rank:
        raise RuntimeError('the abs of begin_params_axis (%d) and begin_norm_axis (%d) '
                           'must be < rank(inputs) (%d)' %
                           (begin_params_axis, begin_norm_axis, in_rank))
    if begin_norm_axis < 0:
        begin_norm_axis = in_rank + begin_norm_axis

    if begin_params_axis < 0:
        begin_params_axis = in_rank + begin_params_axis

    sum_num = reduce(lambda x, y: x * y, shape[begin_norm_axis:])

    # Extracts a slice from a tensor.
    dtype = x.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    ori_dtype = dtype
    if dtype == "float16" and (not utils.product_is_mini()):
        dtype = "float32"
        x = akg.topi.cast(x, dtype)
        dy = akg.topi.cast(dy, dtype)
        variance = akg.topi.cast(variance, dtype)
        mean = akg.topi.cast(mean, dtype)
        gamma = akg.topi.cast(gamma, dtype)

    gamma_reshape = akg.topi.reshape(gamma, shape)
    gamma_brocast = akg.lang.cce.broadcast(gamma_reshape, shape)
    gamma_ = gamma_brocast

    d_x = akg.tvm.compute(shape, lambda *indice: dy(*indice) * gamma_(*indice), name="d_x")

    # cal dvariance
    eps = akg.tvm.const(1e-5, dtype=dtype)
    va_add = akg.tvm.compute(shape, lambda *indice: variance(*indice) + eps, name="va_add_eps")
    va_log = akg.tvm.compute(shape, lambda *indice: akg.tvm.log(va_add(*indice)), name="va_log")
    power_num = akg.tvm.const(-1.5, dtype=dtype)
    va1 = akg.tvm.compute(shape, lambda *indice: va_log(*indice) * power_num, name="va1")
    va3 = akg.tvm.compute(shape, lambda *indice: akg.tvm.exp(va1(*indice)), name="va3")
    neg_half = akg.tvm.const(-0.5, dtype=dtype)
    va = akg.tvm.compute(shape, lambda *indice: neg_half * va3(*indice), name="va4")
    x_minus_mean = akg.tvm.compute(shape, lambda *indice: x(*indice) - mean(*indice), name="x_minus_mean")
    x_minus_mean2 = akg.tvm.compute(shape, lambda *indice: x(*indice) - mean(*indice), name="x_minus_mean")
    va_elems = akg.tvm.compute(shape, lambda *indice: d_x(*indice) * x_minus_mean(*indice) * va(*indice), name="va_elems")
    dvariance1 = va_elems
    for i in range(in_rank - begin_norm_axis):
        dvariance1 = akg.topi.sum(dvariance1, axis=i + begin_norm_axis, keepdims=True)
    dvariance = lang.cce.broadcast(dvariance1, shape)
    two = akg.tvm.const(2, dtype=dtype)
    x_minus_mean_2 = akg.tvm.compute(shape, lambda *indice: two * x_minus_mean(*indice), name="x_minus_mean_2")

    # cal dx
    dx2_1 = akg.topi.divide(x_minus_mean_2, akg.tvm.const(sum_num, dtype=dtype))
    dx2 = lang.cce.vmul(dvariance, dx2_1)

    # cal dmean
    half = akg.tvm.const(-0.5, dtype=dtype)
    me_log_half1 = akg.tvm.compute(shape, lambda *indice: va_log(*indice) * half, name="me_log_half1")
    me_log_half = akg.tvm.compute(shape, lambda *indice: akg.tvm.exp(me_log_half1(*indice)), name="me_log_half")
    me_log_half2 = akg.tvm.compute(shape, lambda *indice: akg.tvm.exp(me_log_half1(*indice)), name="me_log_half")
    dx1 = lang.cce.vmul(d_x, me_log_half)
    dx = lang.cce.vadd(dx1, dx2)
    neg_one = akg.tvm.const(-1, dtype=dtype)
    me_elems1 = akg.tvm.compute(shape, lambda *indice: neg_one * me_log_half(*indice), name="me_elems1")
    me_elems2 = lang.cce.vmul(d_x, me_elems1)

    dmean1 = me_elems2
    for i in range(in_rank - begin_norm_axis):
        dmean1 = akg.topi.sum(dmean1, axis=i + begin_norm_axis, keepdims=True)
    dmean1_1 = lang.cce.broadcast(dmean1, shape)

    me_elems3 = akg.tvm.compute(shape, lambda *indice: neg_one * x_minus_mean_2(*indice), name="me_elems3")

    me_elems2_sum1 = me_elems3
    for i in range(in_rank - begin_norm_axis):
        me_elems2_sum1 = akg.topi.sum(me_elems2_sum1, axis=i + begin_norm_axis, keepdims=True)
    me_elems2_sum2 = lang.cce.broadcast(me_elems2_sum1, shape)
    me_elems2_sum = akg.topi.divide(me_elems2_sum2, akg.tvm.const(sum_num, dtype=dtype))

    dmean2 = lang.cce.vmul(dvariance, me_elems2_sum)

    dmean3 = lang.cce.vadd(dmean1_1, dmean2)
    dmean = lang.cce.broadcast(dmean3, shape)

    dx3 = akg.topi.divide(dmean, akg.tvm.const(sum_num, dtype=dtype))
    dx_last = lang.cce.vadd(dx, dx3)
    dx_last_ = akg.tvm.compute(shape, lambda *indice: dx_last(*indice), name="dx_last")

    # cal dgamm
    x_ = lang.cce.vmul(x_minus_mean2, me_log_half2)
    dg_elems = lang.cce.vmul(dy, x_)

    dgamma_tmp = dg_elems
    if (shape in prob_shape.values()):
        for i in range(begin_params_axis - 1, -1, -1):
            dgamma_tmp = akg.topi.sum(dgamma_tmp, axis=i, keepdims=True)
    else:
        for i in range(begin_params_axis):
            dgamma_tmp = akg.topi.sum(dgamma_tmp, axis=i, keepdims=True)
    dgamma = akg.topi.reshape(dgamma_tmp, shape[begin_params_axis:])

    dgamma_ = akg.tvm.compute(shape[begin_params_axis:], lambda *indice: dgamma(*indice), name="dgamma")

    # cal dbeta
    dbeta_tmp = dy
    for i in range(begin_params_axis):
        dbeta_tmp = akg.topi.sum(dbeta_tmp, axis=i, keepdims=True)
    dbeta_ = akg.topi.reshape(dbeta_tmp, shape[begin_params_axis:])
    dbeta = akg.tvm.compute(shape[begin_params_axis:], lambda *indice: dbeta_(*indice), name="dbeta")

    if ori_dtype != dtype:
        dx_last_ = akg.topi.cast(dx_last_, ori_dtype)
        dgamma_ = akg.topi.cast(dgamma_, ori_dtype)
        dbeta = akg.topi.cast(dbeta, ori_dtype)

    return dx_last_, dgamma_, dbeta, attrs
