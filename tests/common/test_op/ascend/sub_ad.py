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

"""operator dsl function: sub_ad"""

import akg
from akg.ops.math import Sub
from akg.utils import custom_tiling as ct_util

sub_ad_set_dim_map = {
}


def sub_ad_set_dim_func(head, a, b):
    key = []
    key.append(tuple(a.shape))
    key.append(tuple(b.shape))
    key.append(a.dtype)
    hash_key = str(tuple(key))

    if hash_key in sub_ad_set_dim_map.keys():
        return ct_util.set_dims(sub_ad_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key


@ct_util.reg_set_dim_func(sub_ad_set_dim_func)
def sub_ad(head, a, b):
    output = Sub(a, b, target='cce')
    _jacs = list(akg.differentiate(output, [a], head))
    return _jacs[0]
