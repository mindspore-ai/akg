# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function: sigmoid_cross_entropy_with_logits_grad"""
import akg
from akg import tvm, topi
from akg.utils import validation_check as vc_util
from akg.ops.math.exp import exp

# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = -1
SCALAR_NEGTIVE_ONE = -1



def sigmoid_cross_entropy_with_logits_grad_compute(predict, target, dout):
    """sigmoid_cross_entropy_with_logits_grad compute implemention"""
    dtype = predict.dtype
    if dtype == "float16":
        predict = topi.cast(predict, "float32")
        target = topi.cast(target, "float32")
        dout = topi.cast(dout, "float32")

    # e^x
    val1 = exp(predict)
    # 1 + e^x
    val2 = topi.add(val1, tvm.const(SCALAR_ONE, dtype="float32"))
    # e^x / (1 + e^x)
    val3 = topi.divide(val1, val2)
    # -target
    val4 = topi.multiply(target, tvm.const(SCALAR_NEGTIVE_ONE, dtype="float32"))
    # e^x / (1 + e^x) -y
    val5 = topi.add(val3, val4)

    result = topi.multiply(val5, dout)

    if dtype == "float16":
        result = topi.cast(result, dtype)
    return result


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def sigmoid_cross_entropy_with_logits_grad(predict, target, dout):
    """
    Calculating sigmoid_cross_entropy_with_logits_grad.

    Args:
        predict (tvm.tensor.Tensor): Tensor of type float16, float32.
        target (tvm.tensor.Tensor): Tensor of type float16, float32.
        dout (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as predict.
    """
    vc_util.check_shape(predict.shape)
    vc_util.check_shape(target.shape)
    vc_util.check_shape(dout.shape)
    vc_util.ops_dtype_check([predict.dtype, target.dtype, dout.dtype], vc_util.DtypeForDavinci.ALL_FLOAT)

    res = sigmoid_cross_entropy_with_logits_grad_compute(predict, target, dout)

    return res
