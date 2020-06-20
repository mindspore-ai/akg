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

"""sparse softmax cross entropy with logits"""
from akg.ops.nn import sparse_softmax_cross_entropy_with_logits as loss
from akg.ops.nn import sparse_softmax_cross_entropy_with_logits_ad as loss_ad

def SparseSoftmaxCrossEntropyWithLogits(features, labels, is_grad=False, sens=1.0):
    """sparse softmax cross entropy with logits"""
    if is_grad:
        return loss_ad.sparse_softmax_cross_entropy_with_logits_ad(labels, features, reduction='mean', grad_scale=sens)
    return loss.sparse_softmax_cross_entropy_with_logits(labels, features, reduction='mean')
