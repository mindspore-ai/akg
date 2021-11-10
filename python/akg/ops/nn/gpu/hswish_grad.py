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

"""HSwishGrad"""
import akg
import akg.topi as topi
import akg.tvm as tvm
import akg.utils as utils

@akg.schedule(topi.cuda.schedule_injective)
def HSwishGrad(y_grad, x, target=utils.CUDA):
    """
    HSwishGrad
    Args:
        y_grad:
        x:

    Returns:

    """
    if target != utils.CUDA:
        raise RuntimeError("the target %s is not supported!" % target)
    shape = x.shape
    res0 = tvm.compute(shape, lambda *i: tvm.if_then_else(x(*i) <= -3, 0, y_grad(*i) * (2 * x(*i) + 3) / 6))
    res6 = tvm.compute(shape, lambda *i: tvm.if_then_else(x(*i) >= 3, y_grad(*i), res0(*i)))
    return res6