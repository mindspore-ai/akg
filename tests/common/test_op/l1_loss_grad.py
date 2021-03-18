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

"""l1_loss_gard"""
import akg.tvm
import akg.topi
from akg.topi.util import get_const_tuple
from akg.utils import kernel_exec as utils



def l1_loss_grad(pre_deriv, inputs, target):
    """
    do backprop for L1 loss (MAE)
    """
    inputs_dtype = inputs.dtype
    target_dtype = target.dtype
    pre_deriv_dtype = pre_deriv.dtype

    # check inputs data types
    check_list = ["float16", "float32"]
    if not inputs_dtype.lower() in check_list:
        raise RuntimeError("inputs only support %s while dtype is %s" % (
            ",".join(check_list), inputs_dtype))

    if not target_dtype.lower() in check_list:
        raise RuntimeError("target only support %s while dtype is %s" % (
            ",".join(check_list), target_dtype))

    if not pre_deriv_dtype.lower() in check_list:
        raise RuntimeError("prev Derivative only support %s while dtype is %s" % (
            ",".join(check_list), pre_deriv_dtype))

    if not get_const_tuple(target.shape) == get_const_tuple(inputs.shape):
        raise RuntimeError(
            "Please ensure inputs have the same size.", target.shape, prediction.shape)

    inputs_dtype_old = inputs_dtype

    if utils.product_is_mini() and inputs_dtype == 'float32':
        inputs = akg.topi.cast(inputs, "float16")
        target = akg.topi.cast(target, "float16")
        inputs_dtype = "float16"

    def grad_dsl(inputs, target, pre_deriv):
        # do roadcast outside, cause tvm need shape check;if shape not fix how to check
        #pre_deriv = akg.topi.broadcast_to(pre_deriv, inputs.shape)
        coefficient = akg.tvm.const(-1.0, dtype=inputs_dtype)
        res = akg.tvm.compute(inputs.shape,
                          lambda *i: akg.tvm.if_then_else(
                              inputs(*i) >= target(*i),
                              pre_deriv(*i), coefficient * pre_deriv(*i))
                          )
        return res

    cur_deriv = grad_dsl(inputs, target, pre_deriv)
    if utils.product_is_mini() and inputs_dtype_old == 'float32':
        cur_deriv = akg.topi.cast(cur_deriv, inputs_dtype_old)
    return cur_deriv
