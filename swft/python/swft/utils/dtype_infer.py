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

from .util import MMAD_SUPPORT_TYPE
from .util import VECTOR_SUPPORT_TYPE

__all__ = [
    "mmad_dtype_infer",
    "bino_dtype_infer",
    "mono_dtype_infer",
    "scalar_dtype_infer",
    "dup_dtype_infer",
    "conv_dtype_infer",
    "concat_dtype_infer",
    "default_dtype_infer",
    "change_dtype_infer",
    "gather_dtype_infer"
]


def mmad_dtype_infer(a_dtype, b_dtype, c_dtype=None, attrs=None):
    if a_dtype not in MMAD_SUPPORT_TYPE:
        raise TypeError("MMAD not support type {}".format(a_dtype))
    if b_dtype not in MMAD_SUPPORT_TYPE[a_dtype]:
        raise TypeError(
            "MMAD not support type ({}, {})".format(a_dtype, b_dtype))
    if c_dtype is not None and MMAD_SUPPORT_TYPE[a_dtype][b_dtype] != c_dtype:
        raise TypeError("MMAD not support type ({}, {}, {})".format(
            a_dtype, b_dtype, c_dtype))
    return MMAD_SUPPORT_TYPE[a_dtype][b_dtype]


def bino_dtype_infer(a_dtype, b_dtype, attrs=None):
    if a_dtype != b_dtype:
        raise TypeError(
            "Vector op type mismatch ({}, {})".format(a_dtype, b_dtype))
    if a_dtype not in VECTOR_SUPPORT_TYPE:
        raise TypeError("Vector op not support type {}".format(a_dtype))
    return a_dtype


def mono_dtype_infer(a_dtype, attrs=None):
    if a_dtype not in VECTOR_SUPPORT_TYPE:
        raise TypeError("Vector op not support type {}".format(a_dtype))
    return a_dtype


def scalar_dtype_infer(a_dtype, f_dtype, attrs=None):
    if a_dtype not in VECTOR_SUPPORT_TYPE:
        raise TypeError("Vector op not support type {}".format(a_dtype))
    if a_dtype != f_dtype:
        raise TypeError(
            "Vector op type mismatch ({}, {})".format(a_dtype, f_dtype))
    return a_dtype


def conv_dtype_infer(a_dtype, attrs=None):
    dtype = attrs["dtype"]
    if a_dtype not in VECTOR_SUPPORT_TYPE:
        raise TypeError("Conv not support type {}".format(a_dtype))
    if dtype not in VECTOR_SUPPORT_TYPE:
        raise TypeError("Conv not support type {}".format(dtype))
    if a_dtype == dtype:
        raise TypeError("Conv type is same.")
    return dtype


def dup_dtype_infer(f_dtype, attrs=None):
    if f_dtype not in VECTOR_SUPPORT_TYPE:
        raise TypeError("VectorDup not support type {}".format(f_dtype))
    return f_dtype


def concat_dtype_infer(*args, attrs=None):
    if len(args) == 0:
        raise TypeError(
            "Concat op input num should be larger than 0.")
    if not all(args[0] == i for i in args):
        raise TypeError(
            "Concat op's inputs' dtype should be the same, but got {}.".format(" ".join(args)))
    return args[0]


def default_dtype_infer(a_dtype, attrs=None):
    return a_dtype


def gather_dtype_infer(*args, attrs=None):
    return args[0]


def change_dtype_infer(a_dtype, attrs=None):
    new_dtype = attrs["new_dtype"]
    if new_dtype is None:
        return a_dtype
    return new_dtype