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

"""operator dsl function: relu6_grad"""

import akg.tvm
from akg.utils import custom_tiling as ct_util
from akg.utils import validation_check as vc_util

relu6_grad_set_dim_map = {
    str(((8, 14, 14, 6), "float32")): ((8, 1), (2, 1), (2, 1), (6, 1)),
}


def relu6_grad_set_dim_func(dy, features):
    shape = [x.value for x in dy.shape]
    hash_key = str((tuple(shape), dy.dtype))
    return ct_util.set_dims_by_key(hash_key, relu6_grad_set_dim_map), hash_key


@ct_util.reg_set_dim_func(relu6_grad_set_dim_func)
def relu6_grad(dy, features):
    """
    Computes Gradients of Rectified Linear 6.

    Args:
        dy (tvm.tensor.Tensor): Tensor of type float16, float32, gradients backpropagated to the Relu6 op, .
        features (tvm.tensor.Tensor): Tensor of type float16, float32, inputs that where passed to the Relu6 op, or its outputs.

    Returns:
        tvm.tensor.Tensor, has same type and shape as features.
    """

    check_list = ["float16", "float32"]
    dtype = features.dtype
    if not dtype in check_list:
        raise RuntimeError("relu6_grad only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = [x.value for x in features.shape]
    vc_util.check_shape(shape)

    def grad_dsl():
        zeros = 0
        res0 = akg.tvm.compute(shape,
                           lambda *i: akg.tvm.if_then_else(
                               features(*i) >= akg.tvm.const(zeros, dtype),
                               features(*i), akg.tvm.const(zeros, dtype)
                           ))
        res6 = akg.tvm.compute(shape,
                           lambda *i: akg.tvm.if_then_else(
                               features(*i) >= akg.tvm.const(6, dtype),
                               akg.tvm.const(zeros, dtype), res0(*i)
                           ))
        res = akg.tvm.compute(shape,
                          lambda *i: akg.tvm.if_then_else(
                              res6(*i) == akg.tvm.const(zeros, dtype),
                              akg.tvm.const(zeros, dtype), dy(*i)
                          ))

        return res

    return grad_dsl()
