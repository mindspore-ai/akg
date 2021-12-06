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

"""relu6"""
import akg
import akg.tvm as tvm
import akg.utils as utils
import akg.topi as topi
from akg.topi import tag

@tvm.tag_scope(tag=tag.ELEMWISE)
def topi_nn_relu6(x):
    """topi nn relu6."""
    return tvm.compute(x.shape, lambda *i: tvm.min(tvm.max(x(*i), tvm.const(0, x.dtype)), tvm.const(6, x.dtype)))

@akg.schedule(topi.cuda.schedule_injective)
def ReLU6(x, target=utils.CUDA):
    """
    Compute elementwise with function: min(max(x, 0), 6).

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has same type and shape as input.
    
    Supported Platforms:
        'GPU'
    """
    if target != utils.CUDA:
        raise RuntimeError("the target %s is not supported!" % target)
    return topi_nn_relu6(x)
