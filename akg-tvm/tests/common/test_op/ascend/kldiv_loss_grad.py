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

"""operator dsl function: kldiv_loss_grad"""
import akg.topi
from akg.topi.util import get_const_tuple
import akg.utils as utils
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor)
def kldiv_loss_grad(pre_deriv, inputs, outputs):
    """
    do backprop for kldiv loss

    Args:
        pre_deriv (tvm.tensor.Tensor): Gradient tensor for forward output.
        inputs (tvm.tensor.Tensor): Forward input tensor.
        outputs (tvm.tensor.Tensor): Forward output tensor.

    Returns:
        Gradient tensor for forward input.
    """
    inputs_dtype = inputs.dtype
    target_dtype = outputs.dtype
    pre_deriv_dtype = pre_deriv.dtype
    utils.ops_dtype_check([inputs_dtype, target_dtype, pre_deriv_dtype],
                            utils.DtypeForDavinci.ALL_FLOAT)

    if get_const_tuple(outputs.shape) != get_const_tuple(inputs.shape):
        raise RuntimeError("Please ensure inputs have the same size."
                           "", outputs.shape, inputs.shape)

    inputs_dtype_old = inputs_dtype

    if product_is_mini() and inputs_dtype == 'float32':
        inputs = akg.topi.cast(inputs, "float16")
        outputs = akg.topi.cast(outputs, "float16")
        inputs_dtype = "float16"

    cur_deriv = akg.topi.divide(outputs, inputs)
    cur_deriv = akg.topi.multiply(cur_deriv, pre_deriv)
    if product_is_mini() and inputs_dtype_old == 'float32':
        cur_deriv = akg.topi.cast(cur_deriv, inputs_dtype_old)
    return cur_deriv
