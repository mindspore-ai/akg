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

"""operator dsl function:triplet_loss_grad"""

import akg.tvm
from test_op.triplet_loss import triplet_loss_naive
from akg.utils.format_transform import get_shape


def triplet_loss_naive_grad(anchor_output, positive_output, negative_output, grad, margin=1.0):
    """
    Calculate gradient for triplet loss.

    Args:
        anchor_output: Tensor. The training data.
        positive_output: Tensor. Positive samples.
        negative_output: Tensor. Negative samples.
        grad: Tensor.
        margin: Float. Margin for triplet.

    Returns:
        Tensor.
    """
    fwd = triplet_loss_naive(anchor_output, positive_output, negative_output, margin)
    d_pos = (anchor_output - positive_output)
    d_neg = (anchor_output - negative_output)
    an_shape = get_shape(anchor_output)
    zero = akg.tvm.const(0, dtype=anchor_output.dtype)
    d_anchor = akg.tvm.compute(an_shape, lambda i, j: grad[i] * (akg.tvm.expr.Select(fwd[i] == 0, zero, d_pos[i, j] * 2 - 2 * d_neg[i, j])), name="d_anchor")
    d_positive = akg.tvm.compute(an_shape, lambda i, j: grad[i] * (akg.tvm.expr.Select(fwd[i] == 0, zero, -d_pos[i, j] * 2)), name="d_positive")
    d_negative = akg.tvm.compute(an_shape, lambda i, j: grad[i] * (akg.tvm.expr.Select(fwd[i] == 0, zero, 2 * d_neg[i, j])), name="d_negative")

    return d_anchor, d_positive, d_negative
