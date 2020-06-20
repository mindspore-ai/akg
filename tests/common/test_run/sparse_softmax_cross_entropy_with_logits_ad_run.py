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

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.nn import sparse_softmax_cross_entropy_with_logits_ad
from base import get_rtol_atol
from gen_random import random_gaussian

def np_sparse_softmax_cross_entropy_with_logits_ad(shape1, dtype1, shape2, dtype2, reduction="mean", scale=1.0):
    logits = random_gaussian(shape2, miu=0, sigma=1).astype(dtype2)
    num_class = logits.shape[1]
    labels = np.random.randint(low=0, high=num_class, size=shape1).astype(dtype1)
    features = logits
    features_reshape = np.reshape(features, [-1, num_class])
    labels_reshape = np.reshape(labels, [-1])
    batch_dim = 0
    class_dim = 1
    batch_size = features_reshape.shape[batch_dim]
    e = np.exp(features_reshape - np.reshape(
        np.amax(
            features_reshape, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    labels_mat = np.zeros_like(probs).astype(probs.dtype)
    labels_mat[np.arange(batch_size), labels_reshape] = 1.0
    bp = (probs - labels_mat)
    cost = -np.sum(labels_mat * np.log(probs + 1.0e-20), axis=1)
    bp_res = np.reshape(bp, features.shape)
    if not reduction or reduction.lower() == "none":
        bp_res = bp_res
    elif reduction.lower() == "mean":
        cost_num = 1
        for i in range(len(cost.shape)):
            cost_num *= cost.shape[i]
        bp_res = np.divide(bp_res, cost_num)
    elif reduction.lower() == "sum":
         bp_res = bp_res
    else:
        raise ValueError("reduction method for {} is not supported")
    return labels, logits, bp_res*scale


def gen_data(dtype1, dtype2, reduction, shape1, shape2, scale=1.0):
    labels, logits, bp_res = np_sparse_softmax_cross_entropy_with_logits_ad(shape1, dtype1, shape2, dtype2, reduction, scale=scale)
    expect = bp_res
    output_shape = expect.shape
    output = np.full(output_shape, np.nan, dtype2)
    return expect, labels, logits, output


def sparse_softmax_cross_entropy_with_logits_ad_run(shape1, dtype1, shape2, dtype2, reduction, kernel_name, scale=1.0, attrs=None):
    expect, labels, logits, output = gen_data(dtype1, dtype2, reduction, shape1, shape2, scale=scale)
    op_attrs = [reduction]

    op_attrs = op_attrs + [scale]
    mod = utils.op_build_test(sparse_softmax_cross_entropy_with_logits_ad.sparse_softmax_cross_entropy_with_logits_ad,
                              [shape1, shape2], [dtype1, dtype2], op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs)

    output = utils.mod_launch(mod, (labels, logits, output), expect=expect)
    rtol, atol = get_rtol_atol("sparse_softmax_cross_entropy_with_logits_ad", dtype2)
    compare_res = compare_tensor(output, expect, rtol, atol)
    return (labels, logits), output, expect, compare_res
