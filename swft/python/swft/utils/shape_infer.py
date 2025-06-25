#!/usr/bin/env python3
# coding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

from copy import deepcopy

__all__ = [
    "mmad_shape_infer",
    "bino_shape_infer",
    "reduce_shape_infer",
    "brcb_shape_infer",
    "reshape_shape_infer",
    "transpose_shape_infer",
    "split_shape_infer",
    "concat_shape_infer",
    "dup_shape_infer",
    "pad_shape_infer",
    "slice_size_infer",
    "slice_begin_infer",
    "gather_shape_infer",
    "change_shape_infer",
    "default_shape_infer"
]


def mmad_shape_infer(a_shape, b_shape, c_shape=None, attrs=None):
    if (len(a_shape) < 2 or len(b_shape) < 2):
        raise ValueError(
            "MMAD not support shape with {} and {}.".format(a_shape, b_shape))
    if (a_shape[-1] != b_shape[-2]):
        raise ValueError(
            "MMAD shape not match between {} and {},".format(a_shape, b_shape))
    min_len = min(len(a_shape), len(b_shape))
    for i in range(min_len - 2):
        if (a_shape[i - min_len] != b_shape[i - min_len]):
            raise ValueError(
                "MMAD shape not match between {} and {},".format(a_shape, b_shape))
    out_shape = []
    for i in range(len(a_shape) - min_len):
        out_shape.append(a_shape[i])
    for i in range(len(b_shape) - min_len):
        out_shape.append(b_shape[i])
    for i in range(min_len - 1):
        out_shape.append(a_shape[i - min_len])
    out_shape.append(b_shape[-1])
    if (c_shape is not None and out_shape != c_shape):
        raise ValueError("MMAD shape not match among {}, {}, and {},".format(
            a_shape, b_shape, c_shape))
    return out_shape


def bino_shape_infer(a_shape, b_shape, attrs=None):
    min_len = min(len(a_shape), len(b_shape))
    for i in range(min_len):
        if (a_shape[i - min_len] != b_shape[i - min_len]
            and a_shape[i - min_len] % b_shape[i - min_len] != 0
                and b_shape[i - min_len] % a_shape[i - min_len] != 0):
            raise ValueError(
                "Vector op shape mismatch between {} and {}.".format(a_shape, b_shape))
    out_shape = []
    for i in range(len(a_shape) - min_len):
        out_shape.append(a_shape[i])
    for i in range(len(b_shape) - min_len):
        out_shape.append(b_shape[i])
    for i in range(min_len):
        out_shape.append(max(a_shape[i - min_len], b_shape[i - min_len]))
    return out_shape


def reduce_shape_infer(a_shape, attrs):
    reduce_axis = attrs["reduce_axis"][0]
    new_shape = deepcopy(a_shape)
    if reduce_axis >= len(new_shape) or reduce_axis < -len(new_shape):
        raise ValueError("Reduce axis exceed the range.")
    new_shape[reduce_axis] = 1
    return new_shape


def brcb_shape_infer(a_shape, attrs):
    broadcast_axis = attrs["broadcast_axis"][0]
    broad_size = attrs["broad_size"][0]
    new_shape = deepcopy(a_shape)
    if broadcast_axis >= len(new_shape) or broadcast_axis < -len(new_shape):
        raise ValueError("Brcb axis exceed the range.")
    if a_shape[broadcast_axis] != 1:
        raise ValueError("Brcb axis size is not 1.")
    new_shape[broadcast_axis] = broad_size
    return new_shape


def dup_shape_infer(a_shape=None, attrs=None):
    new_shape = attrs["size"]
    return new_shape


def reshape_shape_infer(a_shape, attrs=None):
    total_size = 1
    for x in a_shape:
        total_size *= x
    new_shape = attrs["new_shape"]
    if new_shape is None:
        return a_shape
    reshape_size = 1
    for x in new_shape:
        reshape_size *= x
    if total_size != reshape_size:
        raise ValueError(
            "Reshape shape mismatch between {} and {}.".format(a_shape, new_shape))
    return new_shape


def change_shape_infer(a_shape, attrs=None):
    new_shape = attrs["new_shape"]
    if new_shape is None:
        return a_shape
    return new_shape


def gather_shape_infer(a_shape, b_shape, attrs=None):
    new_shape = []
    batchdims = attrs["batchdims"][0]
    axis = attrs["axis"][0]
    if len(a_shape) != 2:
        raise ValueError("This scenario is not supported now.")
    if (axis != 0) and (axis != 1):
        raise ValueError("This scenario is not supported now.")
    if (batchdims == 1):
        if (axis == 0):
            raise ValueError("This scenario is not supported now.")
        if (axis == 1):
            if (a_shape[0] != b_shape[0]):
                raise ValueError("gather axis size not match.")
            new_shape = b_shape
    if (batchdims != 1):
        raise ValueError("This scenario is not supported now.")
    return new_shape


def transpose_shape_infer(a_shape, attrs):
    new_shape = []
    permute_axis = attrs["permute_axis"]
    if (len(a_shape) != len(permute_axis)):
        raise ValueError("Transpose axis size not match.")
    for x in permute_axis:
        if (x >= len(a_shape) or x < -len(a_shape)):
            raise ValueError("Transpose axis exceed the range.")
    for x in permute_axis:
        new_shape.append(a_shape[x])
    return new_shape


def split_shape_infer(a_shape, attrs=None):
    input_start = attrs["input_start"]
    input_end = attrs["input_end"]
    strides = attrs["strides"]
    if strides is None:
        strides = [1] * len(input_start)
    if input_start is not None:
        if len(input_start) != len(a_shape) or len(input_end) != len(a_shape):
            raise ValueError("Split index not match")
        new_shape = []
        for i in range(len(a_shape)):
            if (input_start[i] >= a_shape[i] or input_end[i] <= input_start[i] or input_end[i] > a_shape[i]):
                raise ValueError("Split index not match")
            new_shape.append(
                (input_end[i] - input_start[i] + strides[i] - 1) / strides[i])
        if "transpose" in attrs:
            new_shape[-2], new_shape[-1] = new_shape[-1], new_shape[-2]
        return new_shape
    return a_shape


def concat_shape_infer(*args, attrs=None):
    a_shape_lst = [j for i in args for j in i]
    output_start = attrs["output_start"]
    output_end = attrs["output_end"]
    out_shape = attrs["out_shape"]
    strides = attrs["strides"]
    if output_start is not None:
        l = len(out_shape)
        if len(output_start) != len(output_end) or len(output_start) != len(a_shape_lst):
            raise ValueError("Concat index not match")
        for i in range(len(a_shape_lst)):
            if ((output_end[i] - output_start[i] + strides[i] - 1) / strides[i] != a_shape_lst[i]):
                raise ValueError("Concat index not match")
            if (output_end[i] <= output_start[i] or output_end[i] > out_shape[i % l]):
                raise ValueError("Concat index not match")
        return out_shape
    return a_shape_lst


def pad_shape_infer(src_shape, pad_shape, attrs):
    pad_row = attrs["pad_row"][0]
    if pad_row != 1:
        raise NotImplementedError("Currently only support row pad")
    if len(src_shape) != len(pad_shape):
        raise ValueError(
            "for ub padding, pad shape should have same dimensions as original shape")
    if src_shape[-1] != pad_shape[-1]:
        raise ValueError("column pad not supported")
    return pad_shape


def slice_size_infer(slice_size):
    if not isinstance(slice_size, (tuple, list)):
        raise TypeError("slice_size should be either tuple or list")
    for ele in slice_size:
        if not isinstance(ele, int):
            raise TypeError("each element in slice size should be integer")
    return slice_size


def slice_begin_infer(slice_begin):
    if not isinstance(slice_begin, (tuple, list)):
        raise TypeError("slice_begin should be either tuple or list")
    return slice_begin


def default_shape_infer(a_shape, attrs=None):
    return deepcopy(a_shape)
