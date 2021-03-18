# Copyright 2021 Huawei Technologies Co., Ltd
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
from akg.utils import validation_check as vc_util

def batch_matmul(data1, data2, bias=None, out_dtype="float32", layout1="NHDT", layout2="NHDT", layout_out="NHDT"):
    if len(data1.shape) == 4:
        res = batch_matmul_4D(data1, data2, bias, out_dtype, layout1, layout2, layout_out)
    elif len(data1.shape) == 2:
        res = batch_matmul_2D(data1, data2, bias, out_dtype, layout1, layout2, layout_out)
    else:
        res = batch_matmul_3D(data1, data2, bias, out_dtype, layout1, layout2, layout_out)
    return res

def auto_out_transpose(expect, layout_out="NHDT"):
    if len(expect.shape) == 3:
        layout_out = layout_out[1:]
    if len(expect.shape) == 2:
        layout_out = layout_out[2:]
    layout_out_int = layout_out.replace('N', '0').replace('H', '1').replace('D', '2').replace('T', '3')
    layout_out_list = list(layout_out_int)
    layout_out_axis = np.argsort(layout_out_list)
    expect = topi.transpose(expect, axes=tuple(layout_out_axis))
    return expect

def batch_matmul_3D(data1, data2, bias=None, out_dtype="float32", layout1="NHDT", layout2="NHDT", layout_out="NHDT"):
    layout1_dict = {}
    layout2_dict = {}
    layout1 = layout1[1:]
    layout2 = layout2[1:]
    layout1_str = layout1.replace('N', 'b').replace('H', 'b').replace('D', 'm').replace('T', 'k')  
    layout2_str = layout2.replace('N', 'b').replace('H', 'b').replace('D', 'n').replace('T', 'k')   
    layout1_list = list(layout1_str)
    layout2_list = list(layout2_str)

    for i in range(len(layout1)):
        layout1_dict[layout1_list[i]] = data1.shape[i]
        layout2_dict[layout2_list[i]] = data2.shape[i]

    reduce_axis = tvm.reduce_axis((0, layout1_dict['k']), name='reduce_axis')

    if out_dtype=="float32":
        res = tvm.compute((layout1_dict['b'], layout1_dict['m'], layout2_dict['n']), lambda b, i, j: tvm.sum(
                data1[b, i if layout1_list[1] == 'm' else reduce_axis, reduce_axis if layout1_list[2] == 'k' else i].astype("float") *
                data2[b, j if layout2_list[1] == 'n' else reduce_axis, reduce_axis if layout2_list[2] == 'k' else j].astype("float"), axis=reduce_axis))
    else:
        res = tvm.compute((layout1_dict['b'], layout1_dict['m'], layout2_dict['n']), lambda b, i, j: tvm.sum(
                data1[b, i if layout1_list[1] == 'm' else reduce_axis, reduce_axis if layout1_list[2] == 'k' else i] *
                data2[b, j if layout2_list[1] == 'n' else reduce_axis, reduce_axis if layout2_list[2] == 'k' else j], axis=reduce_axis))
    if bias is not None:
        res = topi.add(res, bias)

    if layout_out != "NHDT":
        res = auto_out_transpose(res, layout_out)
    return res

def batch_matmul_4D(data1, data2, bias=None, out_dtype="float32", layout1="NHDT", layout2="NHDT", layout_out="NHDT"):
    layout1_dict = {}
    layout2_dict = {}
    layout1_str = layout1.replace('N', 'B').replace('H', 'b').replace('D', 'm').replace('T', 'k')  
    layout2_str = layout2.replace('N', 'B').replace('H', 'b').replace('D', 'n').replace('T', 'k')   
    layout1_list = list(layout1_str)
    layout2_list = list(layout2_str)

    for i in range(len(layout1)):
        layout1_dict[layout1_list[i]] = data1.shape[i]
        layout2_dict[layout2_list[i]] = data2.shape[i]

    reduce_axis = tvm.reduce_axis((0, layout1_dict['k']), name='reduce_axis')

    if out_dtype == "float32":
        res = tvm.compute((layout1_dict['B'], layout1_dict['b'], layout1_dict['m'], layout2_dict['n']), lambda B, b, i, j: tvm.sum(
                data1[B, b, i if layout1_list[2] == 'm' else reduce_axis, reduce_axis if layout1_list[3] == 'k' else i].astype("float") *
                data2[B, b, j if layout2_list[2] == 'n' else reduce_axis, reduce_axis if layout2_list[3] == 'k' else j].astype("float"), axis=reduce_axis))
    else:
        res = tvm.compute((layout1_dict['B'], layout1_dict['b'], layout1_dict['m'], layout2_dict['n']), lambda B, b, i, j: tvm.sum(
                data1[B, b, i if layout1_list[2] == 'm' else reduce_axis, reduce_axis if layout1_list[3] == 'k' else i] *
                data2[B, b, j if layout2_list[2] == 'n' else reduce_axis, reduce_axis if layout2_list[3] == 'k' else j], axis=reduce_axis))
    
    if bias is not None:
        res = topi.add(res, bias)

    if layout_out != "NHDT":
        res = auto_out_transpose(res, layout_out)
    return res

def batch_matmul_2D(data1, data2, bias=None, out_dtype="float32", layout1="NHDT", layout2="NHDT", layout_out="NHDT"):
    layout1_dict = {}
    layout2_dict = {}
    layout1 = layout1[2:]
    layout2 = layout2[2:]
    layout1_str = layout1.replace('D', 'm').replace('T', 'k')  
    layout2_str = layout2.replace('D', 'n').replace('T', 'k')   
    layout1_list = list(layout1_str)
    layout2_list = list(layout2_str)

    for i in range(len(layout1)):
        layout1_dict[layout1_list[i]] = data1.shape[i]
        layout2_dict[layout2_list[i]] = data2.shape[i]

    reduce_axis = tvm.reduce_axis((0, layout1_dict['k']), name='reduce_axis')

    if out_dtype=="float32":
        res = tvm.compute((layout1_dict['m'], layout2_dict['n']), lambda i, j: tvm.sum(
                data1[i if layout1_list[0] == 'm' else reduce_axis, reduce_axis if layout1_list[1] == 'k' else i].astype("float") *
                data2[j if layout2_list[0] == 'n' else reduce_axis, reduce_axis if layout2_list[1] == 'k' else j].astype("float"), axis=reduce_axis))
    else:
        res = tvm.compute((layout1_dict['m'], layout2_dict['n']), lambda i, j: tvm.sum(
                data1[i if layout1_list[0] == 'm' else reduce_axis, reduce_axis if layout1_list[1] == 'k' else i] *
                data2[j if layout2_list[0] == 'n' else reduce_axis, reduce_axis if layout2_list[1] == 'k' else j], axis=reduce_axis))
    
    if bias is not None:
        res = topi.add(res, bias)

    if layout_out != "NHDT":
        res = auto_out_transpose(res, layout_out)
    return res

