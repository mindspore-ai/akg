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

"""operator dsl function:focalloss_ad"""

import akg.tvm
import akg
from tests.common.test_op import focal_loss
from akg.utils import custom_tiling as ct_util


def focalloss_ad2(labels, logits, gamma=2):
    b, _ = focal_loss.focal_loss(logits, labels, gamma)
    head = akg.tvm.placeholder(b.shape, b.dtype, name="head")
    d_logits, d_labels = akg.differentiate(b, [logits, labels], head)
    return d_labels, d_logits, head


focalloss_ad_set_dim_map = {
}

def focalloss_ad_set_dim_func(head, logits, labels, gamma):
    key = []
    key.append(tuple(logits.shape))
    key.append(logits.dtype)
    key.append(labels.dtype)
    key.append(gamma)
    hash_key = str(tuple(key))

    if hash_key in focalloss_ad_set_dim_map.keys():
        return ct_util.set_dims(focalloss_ad_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key


@ct_util.reg_set_dim_func(focalloss_ad_set_dim_func)
def focalloss_ad(head, logits, labels, gamma):
    """Compute gradient of focalloss operator using automatic differentiate."""
    b, _ = focal_loss.focal_loss(logits, labels, gamma)
    jacs = akg.differentiate(b, [logits], head)
    return jacs[0]
