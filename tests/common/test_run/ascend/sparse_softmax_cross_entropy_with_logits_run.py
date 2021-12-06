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

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.nn.ascend import SparseSoftmaxCrossEntropyWithLogits
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol


def np_sparse_softmax_cross_entropy_with_logits(shape1, dtype1, shape2, dtype2, reduction="mean", scale=1.0):
    logits = random_gaussian(shape2, miu=0, sigma=1).astype(dtype2)
    num_class = logits.shape[1]
    labels = np.random.randint(low=0, high=num_class, size=shape1).astype(dtype1)
    batch_dim = 0
    class_dim = 1
    batch_size = logits.shape[batch_dim]
    e = np.exp(logits - np.reshape(
        np.amax(
            logits, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    labels_one_hot = np.zeros_like(probs).astype(probs.dtype)
    labels_one_hot[np.arange(batch_size), labels] = 1.0
    bp = (probs - labels_one_hot)
    cost = -np.sum(labels_one_hot * np.log(probs + 1.0e-20), axis=1)
    if not reduction or reduction.lower() == "none":
        loss = cost
    elif reduction.lower() == "mean":
        loss = np.mean(cost)
        cost_num = 1
        for i in range(len(cost.shape)):
            cost_num *= cost.shape[i]
        bp = np.divide(bp, cost_num)
    elif reduction.lower() == "sum":
        loss = np.sum(cost)
    else:
        raise ValueError("reduction method for {} is not supported")
    # loss_res = np.reshape(loss, labels.shape)
    bp = np.multiply(bp, scale)
    return labels, logits, loss, bp


def sparse_softmax_cross_entropy_with_logits_run(shape1, dtype1, shape2, dtype2, reduction, kernel_name, attrs):
    op_attrs = [reduction]

    attrs["pragma_disable_whole_component"] = False
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(SparseSoftmaxCrossEntropyWithLogits,
                                  [shape1, shape2], [dtype1, dtype2], op_attrs=op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, labels, logits, output = gen_data(dtype1, dtype2, reduction, shape1, shape2)
            return mod, expect, (labels, logits, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(SparseSoftmaxCrossEntropyWithLogits,
                                  [shape1, shape2], [dtype1, dtype2], op_attrs=op_attrs,
                                  kernel_name=kernel_name, attrs=attrs)
        expect, labels, logits, output = gen_data(dtype1, dtype2, reduction, shape1, shape2)
        output = utils.mod_launch(mod, (labels, logits, output), expect=expect)
        rtol, atol = get_rtol_atol("sparse_softmax_cross_entropy_with_logits", dtype2)
        compare_res = compare_tensor(output, expect, rtol=rtol, atol=atol)
        return (labels, logits), output, expect, compare_res


def gen_data(dtype1, dtype2, reduction, shape1, shape2):
    labels, logits, loss_res, _ = np_sparse_softmax_cross_entropy_with_logits(shape1, dtype1, shape2, dtype2,
                                                                                   reduction)
    expect = loss_res
    output_shape = expect.shape
    if reduction and reduction.lower() != "none":
        output_shape = (1,)
        expect = [expect]
    output = np.full(output_shape, np.nan, dtype2)
    return expect, labels, logits, output
