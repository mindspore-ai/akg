#!/usr/bin/env python3
# coding: utf-8
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

"""operator dsl function: sparse_softmax_cross_entropy_with_logits"""
import akg.topi
import akg.tvm
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils import custom_tiling as ct_util
from akg.ops.array.one_hot import one_hot
from akg.ops.math.mul import mul
from akg.ops.math.reduce_max import reduce_max
from akg.ops.math.sub import sub
from akg.ops.math.neg import neg
from akg.ops.math.exp import exp
from akg.ops.math.sum import sum_value
from akg.ops.math.sum import sum_v2
from akg.ops.math.log import log


def sparse_sf_ce_with_logits_tiling_strategy(tensor, out_shape):
    """Custom tiling strategy for sparse softmax cross entropy op."""
    strategy = list()
    for i in range(len(out_shape)):
        if i != len(out_shape) - 1:
            strategy.append(ct_util.create_constraint_on_tensor(tensor=tensor,
                                                                values=1,
                                                                constraints=ct_util.TileConstraint.FACTOR,
                                                                tensor_pos=i)[0])
        else:
            tot_axis = ct_util.create_constraint_on_tensor(tensor=tensor,
                                                           values="FULL",
                                                           constraints=ct_util.TileConstraint.MAX,
                                                           tensor_pos=i)[0]
            strategy.append(tot_axis)
    return strategy


def sparse_softmax_cross_entropy_with_logits_impl(labels=None, logits=None, reduction='mean', scale=1.0):
    """implement of sparse_softmax_cross_entropy_with_logits and sparse_softmax_cross_entropy_with_logits_grad."""
    # dtype check
    labels_dtype_list = ["int32"]

    if labels.dtype not in labels_dtype_list:
        raise ValueError("lables of sparse_softmax_cross_entropy_with_logits only support %s while dtype is %s" % (
            ",".join(labels_dtype_list), labels.dtype))

    # some instructions of float32 not supported on mini
    if logits.dtype == "float32" and utils.product_is_mini():
        raise RuntimeError("The sparse_softmax_cross_entropy_with_logits operator support only float16 for the logits"
                           + "when platform of product is mini")

    # shape check
    labels_shape = vc_util.get_shape(labels)
    logits_shape = vc_util.get_shape(logits)
    vc_util.check_shape_length_equal("labels", labels_shape, 1)
    vc_util.check_shape_length_equal("logits", logits_shape, 2)
    class_axis = 1
    batch_size, num_class = logits_shape[0], logits_shape[class_axis]
    if labels_shape[0] != logits_shape[0]:
        raise ValueError("The length of the first dimension (batch_size) of lables and logits should be equal,"
                         "but got %s and %s" % (labels_shape[0], batch_size))
    labels_one_hot, _ = one_hot(labels, num_class, dtype=logits.dtype)

    # compute for softmax and cross_entropy
    def softmax_cross_entropy_with_logits(labels, logits, axis, reduction="mean", scale=1.0):
        max_logits = reduce_max(logits, axis, keepdims=True)
        data_sub = sub(logits, max_logits)
        akg.register_variables("minus_max", [logits], data_sub)
        data_exp = exp(data_sub)
        data_expsum, _ = sum_value(data_exp, axis, keepdims=True)
        data_expsum_log = log(data_expsum)
        sub_value = sub(data_sub, data_expsum_log)
        neg_labels = neg(labels)
        cross_entropy = mul(neg_labels, sub_value)
        # backprop: prob - labels, where prob = softmax(logits)
        prob = exp(sub_value)
        backprop = sub(prob, labels)

        if reduction.lower() == "none":
            loss, _ = sum_v2(cross_entropy, axis, keepdims=True)
        elif reduction.lower() == "mean":
            loss, _ = sum_v2(cross_entropy, axis=None)
            factor = logits.shape[0].value
            loss = loss * akg.tvm.const(1 / factor, logits.dtype)
            backprop = backprop * akg.tvm.const(1 / factor, logits.dtype)
        elif reduction.lower() == "sum":
            loss, _ = sum_v2(cross_entropy, axis=None)
        else:
            raise ValueError("reduction method {0} is not supported".format(reduction))
        backprop = akg.topi.multiply(backprop, akg.tvm.const(scale, backprop.dtype))
        return loss, backprop
    cost, backprop = softmax_cross_entropy_with_logits(labels_one_hot, logits,
                                                       axis=class_axis, reduction=reduction, scale=scale)

    strategy = sparse_sf_ce_with_logits_tiling_strategy(labels_one_hot, logits_shape)

    return strategy, cost, backprop


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, str)
def sparse_softmax_cross_entropy_with_logits(labels, logits, reduction='mean'):
    """
    Computes sparse softmax cross entropy between `logits` and `labels`.

    Note:
        Softmax calculation of Logits is done inside the op.

    Args:
        labels (tvm.tensor.Tensor): int32 tensor of shape [batch_size].
                                    Each entry in it  must be an index in `[0, num_classes)`.
        logits (tvm.tensor.Tensor): float32 or float16 tensor of shape [batch_size, num_class].
        reduction (str): Specifies the reduction to apply to the output: 'none' or 'mean' or 'sum'. Default: 'mean'.
            'none': no reduction for the output
            'sum': the sum for the output
            'mean': the mean for the output.

    Returns:
        tvm.tensor.Tensor, has the same dtype as logits.
        If reduction is 'none', shape of the tensor is the same as logits,
        otherwise shape of the tensor is the same as labels.
    """
    vc_util.ops_dtype_check(logits.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    strategy, cost, _ = sparse_softmax_cross_entropy_with_logits_impl(labels, logits, reduction)
    attr_map = {"custom_tiling": strategy}
    return cost, attr_map
