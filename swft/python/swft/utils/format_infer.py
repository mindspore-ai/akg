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
__all__ = [
    "mmad_format_infer",
    "bino_format_infer",
    "dup_format_infer",
    "change_format_infer",
    "move_format_infer",
    "concat_format_infer",
    "default_format_infer",
    "gather_format_infer"
]


def mmad_format_infer(a_format, b_format, c_format=None, attrs=None):
    if (a_format != "NZ" and a_format != "NC0HWC1" and a_format != "FM"):
        raise TypeError(
            "MMAD src0 format should be NZ-like, but got {}.".format(a_format))
    if (b_format != "NZ" and b_format != "NC0HWC1" and b_format != "FT"):
        raise TypeError(
            "MMAD src1 format should be NZ-like, but got {}.".format(b_format))
    if (c_format is not None and c_format != "NZ" and c_format != "NC0HWC1" and c_format != "FM"):
        raise TypeError(
            "MMAD src2 format should be NZ-like, but got {}.".format(c_format))
    if (c_format is not None and a_format != c_format):
        raise TypeError(
            "MMAD src0 and src2 format mismatch between {} and {}.".format(a_format, c_format))
    return a_format


def bino_format_infer(a_format, b_format, attrs=None):
    if (a_format != b_format and (a_format == "ND" or b_format == "NZ")):
        raise TypeError(
            "Vector op format mismatch between {} and {}.".format(a_format, b_format))
    return a_format


def dup_format_infer(a_format=None, attrs=None):
    if a_format is None:
        return "ND"
    return a_format


def change_format_infer(a_format, attrs=None):
    new_format = attrs["new_format"]
    if new_format is None:
        return a_format
    return new_format


def move_format_infer(a_format, attrs=None):
    new_format = attrs["format"]
    if new_format is None:
        return a_format
    return new_format


def concat_format_infer(*args, attrs=None):
    if len(args) == 0:
        raise TypeError(
            "Concat op input num should be larger than 0.")
    if not all(args[0] == i for i in args):
        raise TypeError(
            "Concat op's inputs' format should be the same, but got {}.".format(" ".join(args)))
    return args[0]


def default_format_infer(a_format, attrs=None):
    return a_format


def gather_format_infer(*args, attrs=None):
    return args[0]
