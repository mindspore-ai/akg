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

"""operator dsl function:triplet_loss"""
import akg.topi
import akg.tvm

def triplet_loss_naive(anchor_output, positive_output, negative_output, margin=12.0):
    """
    Calculate triplet loss.

    Args:
        anchor_output: Tensor. The training data.
        positive_output: Tensor. Positive samples.
        negative_output: Tensor. Negative samples.
        margin: Float. Margin for triplet.

    Returns:
        Tensor.
    """
    d_pos = akg.topi.sum((anchor_output - positive_output) * (anchor_output - positive_output), -1)
    d_neg = akg.topi.sum((anchor_output - negative_output) * (anchor_output - negative_output), -1)
    loss = akg.tvm.const(margin, anchor_output.dtype) + d_pos - d_neg
    return akg.topi.nn.relu(loss)
