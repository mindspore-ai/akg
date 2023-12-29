# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: sgd"""

import akg
from akg import tvm, topi
import akg.utils as utils
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils


NUM_ZERO = 0.0

def sgd_compute(parameters, gradient, learning_rate, accum, momentum, stat,
                dampening=0.0, weight_decay=0.0, nesterov=False):
    """sgd compute implementation"""
    dtype = parameters.dtype
    if dtype == "float16":
        parameters = topi.cast(parameters, "float32")
        accum = topi.cast(accum, "float32")
        learning_rate = topi.cast(learning_rate, "float32")
        gradient = topi.cast(gradient, "float32")
        momentum = topi.cast(momentum, "float32")
        stat = topi.cast(stat, "float32")

    # if weight_decay != 0.0, need compute grad_delta to update gradient
    if weight_decay != 0.0:
        parameters = topi.multiply(parameters, tvm.const(1.0, 'float32'))
        grad_delta = topi.multiply(parameters, weight_decay)
        gradient = topi.add(gradient, grad_delta)

    stat_mid = topi.multiply(stat, tvm.const(-1, "float32"))
    stat_act = topi.add(stat_mid, tvm.const(1, "float32"))

    dampening_t = topi.multiply(stat_act, dampening)

    # update accum
    accum_delta = tvm.compute(accum.shape, lambda *indice: accum(*indice) * momentum[0])

    gradient_damp = topi.multiply(gradient, dampening_t)
    accum_t = topi.add(accum_delta, gradient)
    if dampening != 0.0:
        accum_t = topi.subtract(accum_t, gradient_damp)

    # update parameters
    if nesterov:
        parameters_delta = tvm.compute(gradient.shape, lambda *indice: gradient(*indice) * learning_rate[0])
        parameters_delta_2 = tvm.compute(accum_t.shape, lambda *indice: accum_t(*indice) * momentum[0])
        parameters_delta_2 = tvm.compute(parameters_delta_2.shape,
                                         lambda *indice: parameters_delta_2(*indice) * learning_rate[0])
        parameters_delta = topi.add(parameters_delta, parameters_delta_2)
        parameters_t = topi.subtract(parameters, parameters_delta)
    else:
        parameters_delta = tvm.compute(accum_t.shape, lambda *indice: accum_t(*indice) * learning_rate[0])
        parameters_t = topi.subtract(parameters, parameters_delta)

    # update stat
    stat_t = topi.multiply(stat_act, tvm.const(NUM_ZERO, 'float32'))


    if dtype == "float16":
        parameters_t = topi.cast(parameters_t, "float16")
        accum_t = topi.cast(accum_t, "float16")
        stat_t = topi.cast(stat_t, "float16")

    return parameters_t, accum_t, stat_t


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (float, type(None)), (float, type(None)),
                          (bool, type(None)))
def sgd(parameters, gradient, accum, stat, learning_rate, momentum, dampening=0.0, weight_decay=0.0, nesterov=False):
    """
    Update parameters, accum and stat according to the SGD algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        parameters -= grad * lr + accum * momentum * lr
    else:
        parameters -= accum * lr

    Args:
        parameters (tvm.tensor.Tensor): parameters tensor of float32, float16, to be updated.
        gradient (tvm.tensor.Tensor): gradient tensor of float32, float16.
        accum (tvm.tensor.Tensor): accum tensor of float32, float16, to be updated.
        stat (tvm.tensor.Tensor): stat tensor of float32, float16, to be updated.
        momentum (tvm.tensor.Tensor): momentum tensor of float32, float16, shape must be equal to (1,).
        learning_rate (tvm.tensor.Tensor): learning_rate tensor of float32, float16, shape must be equal to (1,).
        dampening (float): Default value is 0.0.
        weight_decay (float): Default value is 0.0.
        nesterov (bool): Default is False.

    Return:
        accum_t (tvm.tensor.Tensor): updated accum with same type and shape as accum.
        stat_t (tvm.tensor.Tensor): updated stat with same type and shape as stat.
        parameters_t (tvm.tensor.Tensor): updated parameters with same type and shape as parameters.

    """
    if nesterov and dampening != 0:
        raise ValueError("Nesterov requires zero dampening!")
    if weight_decay < 0:
        raise ValueError("weight_decay must > 0.")

    # shape check
    utils.elemwise_shape_check(parameters.shape, gradient.shape)
    utils.elemwise_shape_check(parameters.shape, accum.shape)
    utils.elemwise_shape_check(parameters.shape, stat.shape)

    # dtype check
    utils.ops_dtype_check([parameters.dtype, gradient.dtype, accum.dtype, stat.dtype],
                            utils.DtypeForDavinci.ALL_FLOAT)

    parameters_t, accum_t, stat_t = sgd_compute(parameters, gradient, learning_rate, accum, momentum, stat, dampening,
                                                weight_decay, nesterov)
    parameters_t, binds_info = TensorUtils.inplace_set(parameters, parameters_t, "parameters_buf")
    accum_t, binds_info2 = TensorUtils.inplace_set(accum, accum_t, "accum_buf")
    stat_t, binds_info3 = TensorUtils.inplace_set(stat, stat_t, "stat_buf")
    binds_info.update(binds_info2)
    binds_info.update(binds_info3)
    attrs = {utils.BINDS: binds_info}


    return parameters_t, accum_t, stat_t, attrs