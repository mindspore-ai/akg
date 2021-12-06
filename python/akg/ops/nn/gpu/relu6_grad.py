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

"""relu6 grad"""
import akg
import akg.topi as topi
import akg.tvm as tvm
import akg.utils as utils

@akg.schedule(topi.cuda.schedule_injective)
def ReLU6Grad(y_grad, x, target=utils.CUDA):
    """
    Computes Gradients of Rectified Linear 6.

    Args:
        y_grad (tvm.tensor.Tensor): Tensor of type float16, float32, gradients backpropagated to the ReLU6 op.
        x (tvm.tensor.Tensor): Tensor of type float16/float32, inputs that where passed to the ReLU6 op, or its outputs.

    Returns:
        tvm.tensor.Tensor, has same type and shape as x.
    
    Supported Platforms:
        'GPU'
    """
    if target != utils.CUDA:
        raise RuntimeError("the target %s is not supported!" % target)
    shape = x.shape
    dtype = x.dtype

    zero = tvm.const(0, dtype)
    six = tvm.const(6, dtype)

    res0 = tvm.compute(shape, lambda *i: tvm.if_then_else(x(*i) >= zero, x(*i), zero))
    res6 = tvm.compute(shape, lambda *i: tvm.if_then_else(x(*i) >= six, zero, res0(*i)))
    res = tvm.compute(shape, lambda *i: tvm.if_then_else(res6(*i) == zero, zero, y_grad(*i)))
    return res
