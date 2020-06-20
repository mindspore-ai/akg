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

"""operator dsl function: softmaxcrossentropywithlogits"""

import akg.tvm
import akg
from akg.utils import validation_check as vc_util


def softmaxcrossentropywithlogits(labels, logits, axis=-1):
    """
    Computes sparse softmax cross entropy between logits and labels.
    cost = labels * -log(softmax(logits))

    Args:
        labels(akg.tvm.Tensor): Tensor of shape [batch_size, num_classes] and dtype float16 or float32.
        logits(akg.tvm.Tensor): Tensor of the same shape and type as labels. Per-label activations.
        axis(int): The class dimension. Default -1, which is the last dimension.

    Returns:
        loss(akg.tvm.Tensor): Tensor of the same type as labels,
                              with shape as labels expect axes specified in axis updated to 1.
        grad(akg.tvm.Tensor): Tennsor of same shape and type as labels.
    """

    shape = [x.value for x in labels.shape]
    vc_util.check_shape(shape)
    if 2 != len(shape):
        raise RuntimeError("Only support rank 2 tensor")
    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        raise ValueError("axis should be less than dimension")
    if axis != len(shape) - 1:
        raise ValueError("Only support the last axis currently")

    # softmax computation
    max_logits = akg.lang.cce.reduce_max(logits, axis=axis)
    max_broadcast = akg.lang.cce.broadcast(max_logits, shape)
    data_sub = akg.lang.cce.vsub(logits, max_broadcast)
    data_exp = akg.lang.cce.vexp(data_sub)
    data_expsum = akg.lang.cce.sum(data_exp, axis)
    data_expsum_broadcast = akg.lang.cce.broadcast(data_expsum, shape)
    para = akg.lang.cce.vrec(data_expsum_broadcast)
    predict = akg.lang.cce.vmul(data_exp, para)

    # cross entropy computation
    predict_log = akg.lang.cce.vlog(predict)
    cross_entropy = akg.lang.cce.vmul(labels, predict_log)
    loss_sum = akg.lang.cce.sum(cross_entropy, -1)

    # compute the loss and the grad
    loss = akg.lang.cce.vmuls(loss_sum, -1)
    deriv = akg.tvm.compute(shape, lambda *index: predict(*index) - labels(*index), name="grad")
    loss_broadcast = akg.lang.cce.broadcast(loss, shape)
    grad = akg.lang.cce.vmul(deriv, loss_broadcast)

    return loss, grad
