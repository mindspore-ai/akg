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

"""operator dsl function: apply_momentum"""
import akg.tvm
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (bool, type(None)), (float, type(None)), (str, type(None)))
def ApplyMomentum(weight, grad, accum, lr_mat, momt_mat, use_nesterov=False, grad_scale=1.0, target=utils.CCE):
    """
    Apply momentum operator.

    Note:
        apply mometum is an op with inplace computing and binds is used.

    Args:
        weight (tvm.tensor.Tensor): weight tensor to be updated.
        grad (tvm.tensor.Tensor): gradient tensor.
        accum (tvm.tensor.Tensor): accum tensor to be updated.
        lr_mat (tvm.tensor.Tensor): tensor with shape (1,).
        momt_mat (tvm.tensor.Tensor): momt_mat tensor with shape (1,).
        use_nesterov (bool): Default value is False.
        grad_scale (float): Default value is 1.0

    Returns:
        fake_output: Invalid value, just suit for framework.
        accum_inplace: tvm.tensor.Tensor, updated accum.
        weight_inplace: tvm.tensor.Tensor, updated weight.
        atts: dict.
    """
    shape = [x.value for x in weight.shape]
    # shape check
    utils.elemwise_shape_check(weight.shape, grad.shape)
    utils.elemwise_shape_check(weight.shape, accum.shape)
    # dtype check
    utils.ops_dtype_check([weight.dtype, grad.dtype, accum.dtype], utils.DtypeForDavinci.ALL_FLOAT)

    grad = akg.tvm.compute(shape, lambda * indice: grad(*indice) * akg.tvm.const(grad_scale, grad.dtype), name="grad")
    momt_accum = akg.tvm.compute(shape, lambda *indice: accum(*indice) * momt_mat[0], name="momt_accum")
    accum_inplace = akg.tvm.compute(shape, lambda *indice: momt_accum(*indice) + grad(*indice), name="accum_inplace")

    if not use_nesterov:
        sum_grad = akg.tvm.compute(shape, lambda *indice: accum_inplace(*indice) * lr_mat[0], name="nesterov_lr")
        weight_inplace = akg.tvm.compute(shape, lambda *indice: weight(*indice) -
                                         sum_grad(*indice), name="weight_inplace")
    else:
        weight_inplace = akg.tvm.compute(shape, lambda *indice: weight(*indice) - grad(*indice) * lr_mat[0]
                                         - accum_inplace(*indice) * momt_mat[0] * lr_mat[0], name="weight_inplace")
    weight_inplace, weight_binds_info = TensorUtils.inplace_set(weight, weight_inplace, "data_buf")
    accum_inplace, accum_binds_info = TensorUtils.inplace_set(accum, accum_inplace, "accum_buf")
    binds_info_all = weight_binds_info
    binds_info_all.update(accum_binds_info)
    attrs = {utils.BINDS: binds_info_all}
    fake_output = akg.tvm.compute(shape, lambda *indice: momt_accum(*indice), name="fake_output")
    # The variable fake_ouput is a invalid value, just to suit for framework of ME !
    # The variable weight_inplace is the updated value of weight .
    # The variable accum_inplace is the updated value of accum .
    return fake_output, accum_inplace, weight_inplace, attrs
