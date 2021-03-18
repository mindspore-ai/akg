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

"""operator dsl function: fused_layernorm"""
from functools import reduce

import akg
import akg.topi
import akg.tvm

from akg.utils import custom_tiling as ct_util, validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.dim import DIM

fused_layernorm_set_dim_map = {
}


def fused_layernorm_set_dim_func(data, _gamma, _beta,
                                 begin_norm_axis, begin_params_axis):
    """dim function"""
    shape = [x.value for x in data.shape]
    hash_key = str((tuple(shape), begin_norm_axis, begin_params_axis, data.dtype))
    return ct_util.set_dims_by_key(hash_key, fused_layernorm_set_dim_map), hash_key


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, int, int)
def fused_layernorm(data, gamma, beta, begin_norm_axis, begin_params_axis):
    dim_info, _ = fused_layernorm_set_dim_func(
        data, gamma, beta, begin_norm_axis, begin_params_axis)
    attrs = {DIM: dim_info}

    # check shapes
    shape_x = get_shape(data)
    vc_util.check_shape(shape_x)

    in_rank = len(shape_x)
    # check begin_norm_axis and begin_params_axis
    if abs(begin_norm_axis) >= in_rank or abs(begin_params_axis) >= in_rank:
        raise RuntimeError('the abs of begin_params_axis (%d) and begin_norm_axis (%d) '
                           'must be < rank(inputs) (%d)' %
                           (begin_params_axis, begin_norm_axis, in_rank))
    if begin_norm_axis < 0:
        begin_norm_axis = in_rank + begin_norm_axis
    if begin_params_axis < 0:
        begin_params_axis = in_rank + begin_params_axis

    # check types
    dtype = data.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    # get fused_layernorm compute
    # cal mean
    if dtype == "float16":
        data_sum = akg.tvm.compute(shape_x, lambda *indice: data(*indice).astype("float32"), "cast_0_fp32")
    else:
        data_sum = data

    for i in range(in_rank - begin_norm_axis):
        data_sum = akg.topi.sum(data_sum, axis=i + begin_norm_axis, keepdims=True)

    if dtype == "float16":
        data_sum = akg.tvm.compute(data_sum.shape, lambda *indice: data_sum(*indice).astype("float16"), "cast_0_fp16")

    sum_num = reduce(lambda x, y: x * y, shape_x[begin_norm_axis:])
    data_mean_tmp = akg.topi.divide(data_sum, akg.tvm.const(sum_num, dtype=dtype))
    data_mean = akg.lang.cce.broadcast(data_mean_tmp, shape_x)

    # cal variance
    data_sub = akg.tvm.compute(data_mean.shape, lambda *indice: data(*indice) - data_mean(*indice), name="sub")
    data_square = akg.tvm.compute(data_mean.shape, lambda *indice: data_sub(*indice) * data_sub(*indice), name="square")

    if dtype == "float16":
        data_sum2 = akg.tvm.compute(data_mean.shape, lambda *indice: data_square(*indice).astype("float32"), "cast_1_fp32")
    else:
        data_sum2 = data_square

    for i in range(in_rank - begin_norm_axis):
        data_sum2 = akg.topi.sum(data_sum2, axis=i + begin_norm_axis, keepdims=True)

    if dtype == "float16":
        data_sum2 = akg.tvm.compute(data_sum2.shape, lambda *indice: data_sum2(*indice).astype("float16"), "cast_1_fp16")

    data_variance_tmp = akg.topi.divide(data_sum2, akg.tvm.const(sum_num, dtype=dtype))
    data_variance = akg.lang.cce.broadcast(data_variance_tmp, shape_x)

    # cal y
    eps = akg.tvm.const(1e-5, dtype=dtype)
    data_add = akg.tvm.compute(data_mean.shape, lambda *indice: data_variance(*indice) + eps, name="add")

    data_log = akg.tvm.compute(data_mean.shape, lambda *indice: akg.tvm.log(data_add(*indice)), name="log")
    power_num = akg.tvm.const(-0.5, dtype=dtype)
    data_muls = akg.tvm.compute(data_mean.shape, lambda *indice: data_log(*indice) * power_num, name="muls")
    data_rsqrt = akg.tvm.compute(data_mean.shape, lambda *indice: akg.tvm.exp(data_muls(*indice)), name="exp")

    data_mul = akg.tvm.compute(data_mean.shape, lambda *indice: data_sub(*indice) * data_rsqrt(*indice), name="mul")

    gamma_reshape = akg.topi.reshape(gamma, shape_x)
    gamma_brocast = akg.lang.cce.broadcast(gamma_reshape, shape_x)
    gamma_ = gamma_brocast
    beta_reshape = akg.topi.reshape(beta, shape_x)
    beta_brocast = akg.lang.cce.broadcast(beta_reshape, shape_x)
    beta_ = beta_brocast

    data_mul2 = akg.tvm.compute(data_mean.shape, lambda *indice: data_mul(*indice) * gamma_(*indice), name="mul2")
    data_add2 = akg.tvm.compute(data_mean.shape, lambda *indice: data_mul2(*indice) + beta_(*indice), name="add2")

    return data_add2, data_mean, data_variance, attrs
