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

"""operator dsl function:squeeze_ad"""

import akg
from test_op import squeeze
from akg.utils import custom_tiling as ct_util

squeeze_ad_set_dim_map = {
    str([(16, 1), 1, "int32"]): ([(16, 16), (1, 1)]),
    str([(8, 16, 1), 2, "int32"]): ([(8, 8), (1, 1), (16, 16), (1, 1)]),
    str([(1, 1, 8, 16), 0, "float16"]): ([(1, 1), (1, 1), (8, 8), (16, 16)]),
    str([(8, 1, 16, 16), 1, "float16"]): ([(8, 8), (1, 1), (16, 16), (16, 16)]),
    str([(1, 3, 1, 4, 1), (0, 2), "int32"]): ([(1, 1), (3, 3), (1, 1), (4, 4), (1, 1)]),
}


def squeeze_ad_set_dim_func(head, data, axis):
    """Lookup squeeze_ad_set_dim_map and return the hash_value and hash_key."""
    key = []
    key.append(tuple(data.shape))
    key.append(axis)
    key.append(data.dtype)

    hash_key = str(tuple(key))

    if hash_key in squeeze_ad_set_dim_map.keys():
        return ct_util.set_dims(squeeze_ad_set_dim_map[hash_key]), hash_key
    return "", hash_key


@ct_util.reg_set_dim_func(squeeze_ad_set_dim_func)
def squeeze_ad(head, data, axis):
    output = squeeze.squeeze(data, axis)
    _jacs = list(akg.differentiate(output, [data], head))
    return _jacs[0]
