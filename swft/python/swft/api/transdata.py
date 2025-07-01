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
def nd_to_nz(src):
    if src.format != "ND":
        raise TypeError(
            "NDtoNZ format mismatch, expect {}, but got {}.".format("ND", src.format))
    if src.mem_type != "UB":
        raise TypeError(
            "NDtoNZ only support mem_type UB, but got {}.".format(src.mem_type))

    dst = Tensor("UB", src.dtype, src.shape,
                 format="NZ", multi_core=src.multi_core)
    dst.load(src)
    return dst


@name_tensor
def nz_to_nd(src):
    if src.format != "NZ":
        raise TypeError(
            "NZtoND format mismatch, expect {}, but got {}.".format("NZ", src.format))
    if src.mem_type != "UB":
        raise TypeError(
            "NZtoND only support mem_type UB, but got {}.".format(src.mem_type))

    dst = Tensor("UB", src.dtype, src.shape,
                 format="ND", multi_core=src.multi_core)
    dst.load(src)
    return dst


@name_tensor
def transpose(src, permute_axis):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = transpose_shape_infer(
        src.shape, attrs={"permute_axis": permute_axis})
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Transpose(src, out, attrs={"permute_axis": permute_axis})()
    return out


@name_tensor
def reshape(src, new_shape):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = reshape_shape_infer(
        src.shape, attrs={"new_shape": new_shape})
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Reshape(src, out, attrs={"new_shape": new_shape})()
    return out


@name_tensor
def change_view(src, new_shape=None, new_format=None, new_dtype=None):
    out_mem_type = move_memtype_infer(src.mem_type)
    out_size = change_shape_infer(
        src.shape, attrs={"new_shape": new_shape})
    out_dtype = change_dtype_infer(src.dtype, attrs={"new_dtype": new_dtype})
    out_format = change_format_infer(
        src.format, attrs={"new_format": new_format})
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    ChangeView(src, out, attrs={
               "new_shape": new_shape, "new_format": new_format, "new_dtype": new_dtype})()
    return out


def transpose_to_gm(dst, src, permute_axis):
    Transpose(src, dst, attrs={
              "permute_axis": permute_axis, "mem_type": dst.mem_type})()
    return dst


def nchw_to_nhwc(src):
    pass


def nhwc_to_nchw(src):
    pass


def nchw_to_nc1hwc0(src):
    pass


def nhwc_to_nc1hwc0(src):
    pass


def nchw_to_c1hwc0n(src):
    pass
