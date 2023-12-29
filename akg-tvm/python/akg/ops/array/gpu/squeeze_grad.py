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

"""squeeze grad"""
import akg.topi as topi
import akg.utils as utils
import akg


def gpu_schedule_squeeze_grad(outs):
    """
    gpu schedule SqueezeGrad.

    Args:
        outs (tvm.tensor.Tensor): outputs of compute.

    Returns:
        sch (schedule.Schedule): The created schedule.
    """
    import default_schedule
    return default_schedule.default_schedule(outs)


@akg.schedule(gpu_schedule_squeeze_grad)
def squeeze_grad(y_grad, x_shape):
    """
    Computes gradients for squeeze op.

    Args:
        y_grad (tvm.tensor.Tensor): the gradient needed to be propagation.
        x_shape (Union[list, tuple]): output Tensor shape.

    Returns:
        tvm.tensor.Tensor: output gradient.
    """
    return topi.reshape(y_grad, x_shape)
