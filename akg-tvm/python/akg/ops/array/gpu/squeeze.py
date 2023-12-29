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

"""squeeze"""
import akg.topi as topi
import akg.utils as utils
import akg


@akg.schedule(topi.cuda.schedule_injective)
def squeeze(x, axis=None, target=utils.CUDA):
    """
    Remove the dimensions which have shape size 1.

    Args:
        x (tvm.tensor.Tensor): Tensor, input whose shape is to be squeeze.
        axis (Union[list, tuple, int, None]): specify which size 1 dimension to be removed.

    Returns:
        tvm.tensor.Tensor, has the same type and element as x, but some size 1 dimensions are removed.

    Supported Platforms:
        'GPU'
    """
    if target != utils.CUDA:
        raise RuntimeError("the target %s is not supported!" % target)
    return topi.squeeze(x, axis)
