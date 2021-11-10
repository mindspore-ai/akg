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

"""dsl: crossentropyloss"""
import akg
import akg.utils as utils


def crossentropyloss(labels, logits, axis=-1, target=utils.CCE):
    ##
    # \Computes cross entropy loss between `logits` and `labels`.
    #
    # \f{eqnarray*}{
    #   cost &=& labels * -log(logits)
    # \f}
    #
    # \param labels akg.tvm.Tensor of shape [batch_size, num_classes]  and dtype float16 or float32.
    # Each row of labels is a valid probability distribution along the class dimension. ??? ont-hot
    # \param logits akg.tvm.Tensor of the same shape and dtype as labels. Per-label activations, typically a linear output.
    # \param axis The class dimension.Defaulted to -1 which is the last dimension.
    #
    # \return loss.
    ##

    shape = [x.value for x in labels.shape]

    utils.check_shape(shape)
    if 2 != len(shape):
        raise RuntimeError("Only support rank 2 tensor")
    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        raise RuntimeError("axis should be less than dimension")
    if axis != len(shape) - 1:
        raise RuntimeError("Only support the last axis currently")

    # logistic computation
    #max = akg.lang.ascend.reduce_max(logits, axis = axis)
    #max_broadcast = akg.lang.ascend.broadcast(max, shape)
    #data_sub = akg.lang.ascend.vsub(logits,max_broadcast)
    #data_exp = akg.lang.ascend.vexp(data_sub)
    #data_exp_1= data_exp + akg.tvm.const(1, dtype=shape)
    #para = akg.lang.ascend.vrec(data_exp_1)
    #predict = akg.lang.ascend.vmul(data_exp, para)

    # cross entropy computation
    predict_log = akg.lang.ascend.vlog(logits)
    cross_entropy = akg.lang.ascend.vmul(labels, predict_log)
    loss_sum = akg.lang.ascend.sum(cross_entropy, -1)

    # compute the loss
    loss = akg.lang.ascend.vmuls(loss_sum, -1)

    return loss
