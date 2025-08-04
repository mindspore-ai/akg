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

from swft.core import Tensor, Scalar, name_tensor
from swft.intrinsic import *
from copy import deepcopy
from swft.utils import *


@name_tensor
def move_to_gm(src, no_autopad=False):
    dst = Tensor("GM", src.dtype, src.shape, src.format, src.multi_core)
    if no_autopad:
        dst.load(src, {"no_pad": [1.0]})
    else:
        dst.load(src)
    return dst


@name_tensor
def move_to_scalar(src):
    if len(src.shape) != 1 and eval_ne(src.shape[0], 1):
        raise ValueError("For move to scalar, src tensor's shape must be [1]")
    dst = Scalar(src.dtype)
    dst.load(src)
    return dst


@name_tensor
def move_to_ub(src, dtype=None, relu=False, no_autopad=False):
    if dtype is None:
        dtype = src.dtype
    dst = Tensor("UB", dtype, src.shape, src.format, src.multi_core)
    attrs = {}
    if relu:
        attrs["relu"] = [1.0]
    if no_autopad:
        attrs["no_pad"] = [1.0]
    if relu or no_autopad:
        dst.load(src, attrs)
    else:
        dst.load(src)
    return dst


@name_tensor
def move_to_l1(src):
    dst = Tensor("L1", src.dtype, src.shape, src.format, src.multi_core)
    dst.load(src)
    return dst


@name_tensor
def move_to_l0A(src, Transpose=False, load3d=False, k_w=0, k_h=0):
    if (load3d):
        c_in = src.shape[3]
        h = src.shape[1]
        w = src.shape[2]
        out_size = [src.shape[0], h * w, c_in * k_w * k_h]
        if Transpose:
            attr = {"c_in": [c_in], "k_w": [k_w],
                    "k_h": [k_h], "transpose": [1.0]}
            out_size = [src.shape[0], c_in * k_w * k_h, h * w]
        else:
            attr = {"c_in": [c_in], "k_w": [k_w], "k_h": [k_h]}
        dst = Tensor("L0A", src.dtype, out_size, "FM", src.multi_core)
        dst.load(src, attr)
    else:
        out_size = deepcopy(src.shape)
        if Transpose:
            attr = {"transpose": [1.0]}
            out_size[-1] = src.shape[-2]
            out_size[-2] = src.shape[-1]
        else:
            attr = None
        dst = Tensor("L0A", src.dtype, out_size, src.format, src.multi_core)
        dst.load(src, attr)
    return dst


@name_tensor
def move_to_l0B(src, Transpose=False, load3d=False, k_w=0, k_h=0):
    if (load3d):
        c_in = src.shape[3]
        h = src.shape[1]
        w = src.shape[2]
        out_size = [src.shape[0], h * w, c_in * k_w * k_h]
        if Transpose:
            attr = {"c_in": [c_in], "k_w": [k_w],
                    "k_h": [k_h], "transpose": [1.0]}
            out_size = [src.shape[0], c_in * k_w * k_h, h * w]
        else:
            attr = {"c_in": [c_in], "k_w": [k_w], "k_h": [k_h]}
        dst = Tensor("L0B", src.dtype, out_size, "FM", src.multi_core)
        dst.load(src, attr)
    else:
        out_size = deepcopy(src.shape)
        if Transpose:
            attr = {"transpose": [1.0]}
            out_size[-1] = src.shape[-2]
            out_size[-2] = src.shape[-1]
        else:
            attr = None
        dst = Tensor("L0B", src.dtype, out_size, src.format, src.multi_core)
        dst.load(src, attr)
    return dst


@name_tensor
def move_to_l0C(src, out_shape, multi_core=True):
    for i in range(len(src.shape) - 1):
        if eval_ne(src.shape[i], 1):
            raise ValueError("Broadcast_to_cc needs bias shape.")
    if eval_ne(src.shape[-1], out_shape[-1]):
        raise ValueError(
            "Broadcast_to_cc shape mismatch between {} and {}.".format(src.shape, out_shape))
    out_size = deepcopy(out_shape)
    out_dtype = "FP32"
    if src.dtype == "INT32":
        out_dtype = "INT32"
    dst = Tensor("L0C", out_dtype, out_size, "NZ", multi_core)
    dst.load(src)
    return dst


@name_tensor
def move_scalar_to_ub(src_s, src_ub, index):
    if isinstance(src_s, (int, float)):
        src_s = Scalar(src_ub.dtype, src_s)
    if not isinstance(src_s, Scalar):
        raise ValueError("src_s should be scalar")
    if not isinstance(src_ub, Tensor):
        raise ValueError("src_ub should be tensor")
    if (src_s.dtype != src_ub.dtype):
        raise TypeError("DType of Scalar and Tensor should be same.")
    out_dtype = mono_dtype_infer(src_ub.dtype)
    out_size = deepcopy(src_ub.shape)
    out_format = deepcopy(src_ub.format)
    out_multi_core = deepcopy(src_ub.multi_core)
    dst = Tensor("UB", out_dtype, out_size, out_format, out_multi_core)
    if isinstance(index, int):
        index = Scalar("INT32", index)
    SetValue(dst, src_ub, src_s, index)()
    return dst
