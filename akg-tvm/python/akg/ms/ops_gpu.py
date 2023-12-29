# Copyright 2021 Huawei Technologies Co., Ltd
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

"""GPU operators"""
import akg.utils as utils
import akg.ops.array.gpu as array
import akg.ops.nn.gpu as nn
from akg.ms.utils import reg_op


@reg_op("Squeeze", utils.CUDA)
def squeeze(x, axis=None, target=utils.CUDA):
    """Squeeze"""
    return array.squeeze(x, axis, target)


@reg_op("SqueezeGrad", utils.CUDA)
def squeeze_grad(y_grad, x_shape, target=utils.CUDA):
    """SqueezeGrad"""
    return array.squeeze_grad(y_grad, x_shape, target)


@reg_op("ReLU6", utils.CUDA)
def relu6(x, target=utils.CUDA):
    """ReLU6"""
    return nn.ReLU6(x, target)


@reg_op("ReLU6Grad", utils.CUDA)
def relu6_grad(y_grad, x, target=utils.CUDA):
    """ReLU6Grad"""
    return nn.ReLU6Grad(y_grad, x, target)


@reg_op("HSwish", utils.CUDA)
def h_swish(x, target=utils.CUDA):
    """HSwish"""
    return nn.HSwish(x, target)


@reg_op("HSwishGrad", utils.CUDA)
def h_swish_grad(y_grad, x, target=utils.CUDA):
    """HSwishGrad"""
    return nn.HSwishGrad(y_grad, x, target)


@reg_op("HSigmoid", utils.CUDA)
def h_sigmoid(x, target=utils.CUDA):
    """HSigmoid"""
    return nn.HSigmoid(x, target)


@reg_op("HSigmoidGrad", utils.CUDA)
def h_sigmoid_grad(y_grad, x, target=utils.CUDA):
    """HSigmoidGrad"""
    return nn.HSigmoidGrad(y_grad, x, target)


@reg_op("GatherNd", utils.CUDA)
def gather_nd(data, indices, target=utils.CUDA):
    """GatherNd"""
    return array.gather_nd(data, indices, target)


@reg_op("Gather", utils.CUDA)
def gather(data, indices, axis, target=utils.CUDA):
    """Gather"""
    return array.gather(data, indices, axis, target)


@reg_op("OneHot", utils.CUDA)
def one_hot(indices, on_value, off_value, depth, axis, dtype, target=utils.CUDA):
    """OneHot"""
    return array.one_hot(indices, on_value, off_value, depth, axis, dtype, target)


@reg_op("StandardNormal", utils.CUDA)
def standard_normal(seed, shape, target=utils.CUDA):
    """StandardNormal"""
    return array.standard_normal(seed, shape, target)


@reg_op("TensorScatterAdd", utils.CUDA)
def tensor_scatter_add(data, indices, updates, target=utils.CUDA):
    """TensorScatterAdd"""
    return array.tensor_scatter_add(data, indices, updates, target)
