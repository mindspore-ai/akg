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

"""operator dsl function:relu_grad"""
import akg.tvm
import akg.utils as utils



def relu_grad(inputs, head, target="cce"):
    """
    Computes gradient of inputs for the relu op

    Args:
        inputs: It is the same with the relu op.
        head: Tensor, has the same type and shape as inputs. Back propagation value.

    Returns:
        Tensor, has the same type and shape as inputs.
    """

    check_list = ["float16", "float32"]
    dtype = inputs.dtype
    if not dtype.lower() in check_list:
        raise RuntimeError("relu_grad only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = [x.value for x in inputs.shape]
    utils.check_shape(shape)

    res = akg.tvm.compute(shape,
                          lambda *i: akg.tvm.if_then_else(
                              inputs(*i) > akg.tvm.const(0, dtype),
                              head(*i), akg.tvm.const(0, dtype)
                          ))
    return res
