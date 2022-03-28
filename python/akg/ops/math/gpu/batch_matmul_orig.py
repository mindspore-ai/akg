# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: batch_matmul"""
import numpy as np
import akg.topi as topi
import akg.tvm as tvm
import akg.utils as utils
from .matmul_utils import auto_out_transpose


def batch_matmul(data1, data2, bias=None, layout1="NHDT", layout2="NHDT", layout_out="NHDT"):
    if len(data1.shape) == 4:
        res = batch_matmul_4D(data1, data2, bias, layout1, layout2, layout_out)
    elif len(data1.shape) == 2:
        res = batch_matmul_2D(data1, data2, bias, layout1, layout2, layout_out)
    else:
        res = batch_matmul_3D(data1, data2, bias, layout1, layout2, layout_out)
    return res


def auto_in_transpose(data, layout="NHDT"):
    layout_int = layout.replace('N', '0').replace(
        'H', '1').replace('D', '2').replace('T', '3')
    layout_list = list(layout_int)
    layout_axis = np.argsort(layout_list)
    data = topi.transpose(data, axes=tuple(layout_axis))
    return data


def batch_matmul_3D(data1, data2, bias, layout1="NHDT", layout2="NHDT", layout_out="NHDT"):
    if layout1 != "NHDT":
        layout1 = layout1[1:]
        data1 = auto_in_transpose(data1, layout1)
    if layout2 != "NHDT":
        layout2 = layout2[1:]
        data2 = auto_in_transpose(data2, layout2)
    res = topi.nn.batch_matmul(data1, data2)
    if bias is not None:
        res = topi.add(res, bias)

    if layout_out != "NHDT":
        res = auto_out_transpose(res, layout_out)
    return res


def batch_matmul_4D(data1, data2, bias, layout1="NHDT", layout2="NHDT", layout_out="NHDT"):
    if layout1 != "NHDT":
        data1 = auto_in_transpose(data1, layout1)
    if layout2 != "NHDT":
        data2 = auto_in_transpose(data2, layout2)

    b1, b2, m, k = data1.shape
    b1, b2, n, k = data2.shape
    reduce_axis = tvm.reduce_axis((0, k), name='reduce_axis')
    res = tvm.compute((b1, b2, m, n), lambda i_b1, i_b2, i_m, i_n: tvm.sum(data1[i_b1, i_b2, i_m, reduce_axis] *
                                                                           data2[i_b1, i_b2,
                                                                                 i_n, reduce_axis],
                                                                           axis=reduce_axis), name='matmul_compute')
    if bias is not None:
        res = topi.add(res, bias)
    if layout_out != "NHDT":
        res = auto_out_transpose(res, layout_out)
    return res


def batch_matmul_2D(data1, data2, bias, layout1="NHDT", layout2="NHDT", layout_out="NHDT"):
    if layout1 != "NHDT":
        layout1 = layout1[2:]
        data1 = auto_in_transpose(data1, layout1)
    if layout2 != "NHDT":
        layout2 = layout2[2:]
        data2 = auto_in_transpose(data2, layout2)

    m, k = data1.shape
    n, k = data2.shape
    reduce_axis = tvm.reduce_axis((0, k), name='reduce_axis')
    res = tvm.compute((m, n), lambda i_m, i_n: tvm.sum(data1[i_m, reduce_axis] *
                                                       data2[i_n, reduce_axis],
                                                       axis=reduce_axis), name='matmul_compute')
    if bias is not None:
        res = topi.add(res, bias)
    if layout_out != "NHDT":
        res = auto_out_transpose(res, layout_out)
    return res
