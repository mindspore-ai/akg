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

"""operator dsl function: smooth_l1_loss_ad"""
import akg
import akg.utils as utils
from tests.common.test_op.ascend import smooth_l1_loss
from akg.utils import custom_tiling as ct_util
smooth_l1_loss_ad_set_dim_map = {
    str(((32, 8732, 4), "float16", "int32")): ((1, 1), (236, 236), (4, 4)),
}


def smooth_l1_loss_ad_set_dim_func(head, prediction, tar, anchor_samples, anchor_sample_correct=0, delta=1.0):
    key = []
    for x in prediction.shape:
        key.append(x)

    hash_key = str((tuple(key), prediction.dtype, anchor_samples.dtype))
    return ct_util.set_dims_by_key(hash_key, smooth_l1_loss_ad_set_dim_map), hash_key


@ct_util.reg_set_dim_func(smooth_l1_loss_ad_set_dim_func)
def smooth_l1_loss_ad(head, prediction, tar, anchor_samples, anchor_sample_correct=0, delta=1.0):
    b = smooth_l1_loss.smooth_l1_loss(prediction, tar, anchor_samples, anchor_sample_correct, delta)
    _jacs = list(akg.differentiate(b[0], [prediction], head))
    return _jacs[0]
