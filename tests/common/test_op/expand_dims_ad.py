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

"""operator dsl function: expand_dims_ad"""
import akg
from tests.common.test_op import expand_dims
from akg.utils import custom_tiling as ct_util

expand_dims_ad_set_dim_map = {
    str(((8, 128), 2, "int32")): (((8, 8), (128, 128))),
    str(((64, 128), 2, "int32")): (((64, 64), (128, 128))),
    str(((64, 128, 128), 1, "float16")): (((1, 1), (128, 128), (128, 128))),
    str(((8, 128, 128), 1, "float16")): (((1, 1), (128, 128), (128, 128))),
    str(((8, 128, 128), 1, "float32")): (((1, 1), (128, 128), (128, 128))),
}


def expand_dims_ad_set_dim_func(head, data, axis):
    key = []
    key.append(tuple(data.shape))
    key.append(axis)
    key.append(data.dtype)
    hash_key = str(tuple(key))

    if hash_key in expand_dims_ad_set_dim_map.keys():
        return ct_util.set_dims(expand_dims_ad_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key


@ct_util.reg_set_dim_func(expand_dims_ad_set_dim_func)
def expand_dims_ad(head, data, axis):
    """Compute gradient of expand_dims operator using automatic differentiate."""
    output = expand_dims.expand_dims(data, axis)
    jacs = list(akg.differentiate(output, [data], head))
    return jacs[0]
