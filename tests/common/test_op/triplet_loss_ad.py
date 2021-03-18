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

"""operator dsl function: triplet_loss_ad"""
import akg
from tests.common.test_op.triplet_loss import triplet_loss_naive

def triplet_loss_ad(head, anchor_output, positive_output, negative_output, margin=1.0, input_id=0):
    if not ((input_id >= 0) and (input_id <= 2)):
        raise RuntimeError("Error: input_id should be 0, 1 or 2 only!")
    fwd = triplet_loss_naive(anchor_output, positive_output, negative_output, margin)

    if (input_id == 0):
        _jacs = list(akg.differentiate(fwd, [anchor_output, positive_output, negative_output], head))
    elif (input_id == 1):
        _jacs = list(akg.differentiate(fwd, [positive_output], head))
    else:
        _jacs = list(akg.differentiate(fwd, [negative_output], head))
    return _jacs[0]
