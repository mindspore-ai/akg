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

"""
dropout dsl
"""

import akg.tvm
from akg.lang import ascend as dav
import akg.topi
from akg import dim
from akg.utils import custom_tiling as ct_util
import akg.utils as utils


def dropout_set_dim_func(data_tensor, data_mask, keep_prob):
    shape = [x.value for x in data_tensor.shape if x.value != 1]
    dtype = data_tensor.dtype
    storage = 49152
    if dtype.lower() == 'float16':
        dnum = 1
    else:
        dnum = 2

    info = dim.Dim()
    list_info = []

    def cal_max_divisor(a, threshold):
        for i in range(threshold, 0, -1):
            if a % i == 0:
                return i
        return 1
    for i in range(len(shape) - 1, -1, -1):
        if dnum >= storage:
            list_info.append((i, 1))
        elif dnum * shape[i] > storage:
            list_info.append((i, cal_max_divisor(shape[i], storage // dnum)))
        dnum *= shape[i]

    for i in reversed(list_info):
        info.setdim(index=0, axis=i[0], tilel1=i[1], tilel0=1)

    return str(info)


@ct_util.reg_set_dim_func(dropout_set_dim_func)
def dropout_do_mask(data_tensor, data_mask, keep_prob):
    dtype = data_tensor.dtype
    shape_tensor = [x.value for x in data_tensor.shape]
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(shape_tensor)

    strides = [1]
    for x in reversed(shape_tensor):
        strides.append(strides[-1] * x)

    if keep_prob < 0 or keep_prob > 1:
        raise RuntimeError("keep_prob must in [0,1]")

    keep_prob_const = akg.tvm.const(1.0 / keep_prob, dtype=dtype)
    data_scale_ub = akg.tvm.compute(
        shape_tensor,
        lambda *indices: data_tensor(*indices) * keep_prob_const,
        name='data_scale_ub')

    def get_index(indices):
        idx = 0
        for i in range(len(indices)):
            idx += indices[len(indices) - i - 1] * strides[i]
        return idx // 8

    if dtype == "float32":
        data_scale_ub_16 = akg.topi.cast(data_scale_ub, "float16")
        res_ub_16 = akg.tvm.compute(shape_tensor,
                                lambda *indice: dav.dropout(data_mask[get_index(indice)], data_scale_ub_16(*indice)))
        res = akg.topi.cast(res_ub_16, "float32")
    else:
        res = akg.tvm.compute(shape_tensor, lambda *indice: dav.dropout(data_mask[get_index(indice)], data_scale_ub(*indice)))

    return res
