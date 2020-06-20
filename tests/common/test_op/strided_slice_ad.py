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

"""operator dsl function: stride_slice_ad"""
import akg.tvm
import akg
import akg.lang.cce
from akg.utils import custom_tiling as ct_util


def get_shape(pld): return [d.value for d in pld.shape]


def stridedslice(a, begins, ends, strides):
    a_shape = get_shape(a)

    assert len(begins) == len(a_shape)
    assert len(ends) == len(a_shape)
    assert len(strides) == len(a_shape)
    assert all(b >= 0 and b <= d for d, b in zip(a_shape, begins))
    assert all(e > b and e <= d for d, b, e in zip(a_shape, begins, ends))

    out_shape = [(e - b) // s for b, e, s in zip(begins, ends, strides)]
    return akg.tvm.compute(out_shape,
                       lambda *i: a(*[b + idx * s for b, s, idx in zip(begins, strides, i)]))


strided_slice_ad_set_dim_map = {
    str(([4, 4, 8, 8], [0, 0, 0, 0], [4, 4, 8, 8], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 4, 8, 8], [0, 0, 0, 0], [4, 4, 8, 8], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 4, 8, 8], [1, 1, 0, 0], [3, 3, 8, 8], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 8, 8], [1, 0, 0], [3, 8, 8], [1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 4, 8, 8], [0, 1, 0, 0], [3, 4, 8, 8], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 4, 8, 8], [0, 0, 4, 0], [3, 4, 8, 8], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 4, 8, 8], [0, 1, 0, 0], [3, 4, 4, 8], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 4, 8, 8], [0, 0, 4, 0], [4, 4, 8, 8], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 4, 8, 8], [0, 0, 0, 0], [4, 4, 8, 4], [1, 1, 1, 1], "float16")): ([(1, 1), (1, 1), (1, 1), (4, 4)]),
    str(([4, 8, 8], [0, 0, 0], [4, 8, 4], [1, 1, 1], "float16")): ([(1, 1), (1, 1), (4, 4)]),
    str(([8, 8, 16, 16], [2, 4, 0, 0], [4, 8, 8, 16], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([64, 16, 16], [0, 1, 0], [64, 2, 16], [1, 1, 1], "float16")): ([(1, 1), (1, 1), (16, 16)]),
    str(([32, 32, 16], [0, 0, 0], [32, 1, 16], [1, 1, 1], "float16")): ([(1, 1), (1, 1), (16, 16)]),
    str(([8, 8, 16, 16], [0, 0, 0, 0], [4, 8, 8, 16], [1, 1, 1, 1], "float16")): ([(1, 0)]),
    str(([4, 8, 8], [0, 0, 0], [4, 8, 8], [1, 1, 1], "float16")): ([(1, 1), (1, 1), (4, 4)]),
    str(([4, 8, 8], [0, 0, 0], [4, 8, 8], [1, 1, 1], "float16")): ([(1, 1), (1, 1), (4, 4)]),
    str(([4, 16, 16], [0, 0, 0], [4, 16, 16], [1, 2, 1], "float16")): ([(1, 1), (1, 1), (16, 16)]),
    str(([4, 16, 16], [0, 0, 0], [4, 16, 16], [2, 1, 1], "float16")): ([(1, 1), (1, 1), (16, 16)]),
    str(([4, 16, 16], [0, 0, 0], [4, 16, 16], [2, 2, 1], "float16")): ([(1, 1), (1, 1), (16, 16)]),
    str(([4, 4, 8, 8], [0, 0, 0, 0], [4, 4, 8, 8], [1, 1, 1, 1], "float32")): ([(1, 0)]),
}


def strided_slice_ad_set_dim_func(head, a, begin, end, strides, dtype):
    key = []
    key.append(tuple(a.shape))
    key.append(tuple(begin))
    key.append(tuple(end))
    key.append(tuple(strides))
    key.append(dtype)
    hash_key = str(tuple(key))

    if hash_key in strided_slice_ad_set_dim_map.keys():
        return ct_util.set_dims(strided_slice_ad_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key


@ct_util.reg_set_dim_func(strided_slice_ad_set_dim_func)
def strided_slice_ad(head, a, begin, end, strides, dtype):
    b = stridedslice(a, begin, end, strides)
    _jacs = list(akg.differentiate(b, [a], head))
    return _jacs[0]
