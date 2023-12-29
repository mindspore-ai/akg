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

"""operator dsl function: softmaxcrossentropywithlogits"""

import akg.tvm
import akg
import akg.utils as utils


def softmaxcrossentropywithlogits(labels, logits, axis=-1, target="cce"):
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
    utils.check_shape(shape)
    if 2 != len(shape):
        raise RuntimeError("Only support rank 2 tensor")
    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        raise ValueError("axis should be less than dimension")
    if axis != len(shape) - 1:
        raise ValueError("Only support the last axis currently")

    # softmax computation
    max_logits = akg.lang.ascend.reduce_max(logits, axis=axis)
    max_broadcast = akg.lang.ascend.broadcast(max_logits, shape)
    data_sub = akg.lang.ascend.vsub(logits, max_broadcast)
    data_exp = akg.lang.ascend.vexp(data_sub)
    data_expsum = akg.lang.ascend.sum(data_exp, axis)
    data_expsum_broadcast = akg.lang.ascend.broadcast(data_expsum, shape)
    para = akg.lang.ascend.vrec(data_expsum_broadcast)
    predict = akg.lang.ascend.vmul(data_exp, para)

    # cross entropy computation
    predict_log = akg.lang.ascend.vlog(predict)
    cross_entropy = akg.lang.ascend.vmul(labels, predict_log)
    loss_sum = akg.lang.ascend.sum(cross_entropy, -1)

    # compute the loss and the grad
    loss = akg.lang.ascend.vmuls(loss_sum, -1)
    deriv = akg.tvm.compute(shape, lambda *index: predict(*index) - labels(*index), name="grad")
    loss_broadcast = akg.lang.ascend.broadcast(loss, shape)
    grad = akg.lang.ascend.vmul(deriv, loss_broadcast)

    return loss, grad
