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

"""operator dsl function: elewise_mul_ad"""
import akg.tvm
import akg
import akg.lang.cce
from akg.utils import custom_tiling as ct_util

elemwise_mul_ad_set_dim_map = {
    str(([3, 3], "float16")): ([(1, 0)]),
    str(([3, 3], "float32")): ([(1, 0)]),
}


def elemwise_mul_ad_set_dim_func(head, a, a2):
    key = []
    key.append(tuple(a.shape))
    key.append(a.dtype)
    hash_key = str(tuple(key))

    if hash_key in elemwise_mul_ad_set_dim_map.keys():
        return ct_util.set_dims(elemwise_mul_ad_set_dim_map[hash_key])
    else:
        return ""


@ct_util.reg_set_dim_func(elemwise_mul_ad_set_dim_func)
def elemwise_mul_ad(head, a, a2):
    b = akg.tvm.compute(a.shape, lambda *indices: a(*indices) * a2(*indices), name="b")
    _jacs = list(akg.differentiate(b, [a], head))
    return _jacs[0]
