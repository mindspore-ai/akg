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

"""operator dsl function: sparse_softmax_cross_entropy_with_logits_ad"""
import akg.tvm
import akg
from akg.ops.nn import sparse_softmax_cross_entropy_with_logits as loss
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, str, float)
def sparse_softmax_cross_entropy_with_logits_ad(labels, logits, reduction='mean', grad_scale=1.0):
    """Compute gradient for sparse_softmax_cross_entropy_with_logits operator using automatic differentiate."""
    attr_map = {}

    def custom_softmax_cross_entropy_with_logits_fdiff(out, inputs, grad, attrs, new_pld_array):
        strategy, _, backprop = loss.sparse_softmax_cross_entropy_with_logits_impl(inputs[1], inputs[0],
                                                                                   reduction=reduction,
                                                                                   scale=grad_scale)
        if strategy:
            attr_map["custom_tiling"] = strategy
        return [backprop]
    l_value, _ = loss.sparse_softmax_cross_entropy_with_logits(labels, logits, reduction)
    head = akg.tvm.compute(l_value.shape, lambda *i: akg.tvm.const(1.0, l_value.dtype), name='head')
    [dl_dlogits] = akg.differentiate(l_value, [logits], head, None, None,
                                     override={l_value: ([logits, labels],
                                                         custom_softmax_cross_entropy_with_logits_fdiff)})
    return dl_dlogits, attr_map
