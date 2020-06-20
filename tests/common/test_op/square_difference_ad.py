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

"""operator dsl function: square_difference_ad"""

import akg
from test_op import square_difference
from akg.utils import custom_tiling as ct_util

square_difference_ad_set_dim_map = {
    str(((160, 1024), (160, 1), "float16")): (((1, 1), (1024, 1024))),
    str(((1024, 1024), (1024, 1), "float16")): (((1, 1), (1024, 1024))),
    str(((1280, 1024), (1280, 1), "float16")): (((1, 1), (1024, 1024))),
    str(((8192, 1024), (8192, 1), "float16")): (((1, 1), (1024, 1024))),
    str(((8, 128, 1024), (8, 128, 1), "float16")): (((1, 1), (1, 1), (1024, 1024))),
    str(((64, 128, 1024), (64, 128, 1), "float16")): (((1, 1), (1, 1), (1024, 1024))),
}


def square_difference_ad_set_dim_func(head, a_up, b_up):
    """Lookup the square_difference_ad_set_dim_map and return hash_value and hash_key."""
    key = []
    key.append(tuple(a_up.shape))
    key.append(tuple(b_up.shape))
    key.append(a_up.dtype)
    hash_key = str(tuple(key))

    if hash_key in square_difference_ad_set_dim_map.keys():
        return ct_util.set_dims(square_difference_ad_set_dim_map[hash_key]), hash_key
    return "", hash_key


@ct_util.reg_set_dim_func(square_difference_ad_set_dim_func)
def square_difference_ad(head, a_up, b_up):
    c_up = square_difference.square_difference(a_up, b_up)
    _jacs = list(akg.differentiate(c_up, [a_up], head))
    return _jacs[0]
