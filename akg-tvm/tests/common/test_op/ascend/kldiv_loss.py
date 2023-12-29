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

"""operator dsl function: kldiv_loss"""
import numpy
import akg.topi
from akg.topi.util import get_const_tuple
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          (str, type(None)))
def kldiv_loss(inputs, outputs, reduction='none'):
    """
    Computes Kullback-Leibler divergence loss between outputs and inputs.

    In default, loss = outputs*(log(outputs) - log(inputs)),
    the way using to reduce loss is defined in reduction

    Args:
        inputs (tvm.tensor.Tensor): Tensor with type float16, float32
        outputs (tvm.tensor.Tensor): Tensor with same type as inputs.
        reduction (str): uses one of ['sum', 'mean', 'batchmean']

    Returns:
        Tensor with same type as input tensors.
    """

    inputs_dtype = inputs.dtype
    target_dtype = outputs.dtype
    utils.ops_dtype_check([inputs_dtype, target_dtype],
                            utils.DtypeForDavinci.ALL_FLOAT)

    if get_const_tuple(outputs.shape) != get_const_tuple(inputs.shape):
        raise RuntimeError(
            "Please ensure inputs have the same size.", outputs.shape, inputs.shape)

    inputs_dtype_old = inputs_dtype

    if product_is_mini() and inputs_dtype == 'float32':
        inputs = akg.topi.cast(inputs, "float16")
        outputs = akg.topi.cast(outputs, "float16")
        inputs_dtype = "float16"

    log_inputs = akg.topi.log(inputs)
    log_target = akg.topi.log(outputs)
    loss = akg.topi.subtract(log_target, log_inputs)
    loss = akg.topi.multiply(outputs, loss)
    if reduction == 'sum':
        loss = akg.topi.sum(loss)
    if reduction == 'mean':
        loss = akg.topi.sum(loss)
        deno = 1.0
        for num in inputs.shape:
            deno = deno * num
        deno = akg.topi.cast(deno, dtype=inputs_dtype)
        loss = akg.topi.divide(loss, deno)
    if reduction == 'batchmean':
        reduce_axis = tuple(numpy.arange(1, len(inputs.shape)))
        loss = akg.topi.sum(loss, axis=reduce_axis, keepdims=False)
        deno = 1.0
        for num in inputs.shape[1:]:
            deno = deno * num
        deno = akg.topi.cast(deno, dtype=inputs_dtype)
        loss = akg.topi.divide(loss, deno)

    if product_is_mini() and inputs_dtype_old == 'float32':
        loss = akg.topi.cast(loss, inputs_dtype_old)
    return loss
