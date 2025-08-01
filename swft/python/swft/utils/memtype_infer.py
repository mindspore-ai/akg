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
    "mmad_memtype_infer",
    "bino_memtype_infer",
    "mono_memtype_infer",
    "dup_memtype_infer",
    "move_memtype_infer",
    "concat_memtype_infer",
    "sel_memtype_infer",
    "default_memtype_infer"
]


def mmad_memtype_infer(a_mem, b_mem, c_mem=None, attrs=None):
    if (a_mem != "L0A"):
        raise TypeError(
            "MMAD mem_type not be {}, but got {}.".format("L0A", a_mem))
    if (b_mem != "L0B"):
        raise TypeError(
            "MMAD mem_type not be {}, but got {}.".format("L0B", b_mem))
    out_mem = "L0C"
    if (c_mem is not None and out_mem != c_mem):
        raise TypeError(
            "MMAD mem_type not be {}, but got {}.".format(out_mem, c_mem))
    return out_mem


def bino_memtype_infer(a_mem, b_mem, attrs=None):
    if (a_mem != "UB"):
        raise TypeError(
            "Vector op mem_type not be {}, but got {}.".format("UB", a_mem))
    if (b_mem != "UB"):
        raise TypeError(
            "Vector op mem_type not be {}, but got {}.".format("UB", b_mem))
    return "UB"

def sel_memtype_infer(a_mem, b_mem, cond_mem, attrs=None):
    if (a_mem != "UB"):
        raise TypeError(
            "Vector op mem_type not be {}, but got {}.".format("UB", a_mem))
    if (b_mem != "UB"):
        raise TypeError(
            "Vector op mem_type not be {}, but got {}.".format("UB", b_mem))
    if (cond_mem != "UB"):
        raise TypeError(
            "Vector op mem_type not be {}, but got {}.".format("UB", cond_mem))
    return "UB"



def mono_memtype_infer(a_mem, attrs=None):
    if (attrs is not None):
        if ("mem_type" in attrs):
            return attrs["mem_type"]

    if (a_mem != "UB"):
        raise TypeError(
            "Vector op mem_type not be {}, but got {}.".format("UB", a_mem))
    return a_mem


def dup_memtype_infer(a_mem=None, attrs=None):
    return "UB"


def move_memtype_infer(a_mem, attrs=None):
    if not attrs:
        return a_mem
    if "mem_type" not in attrs:
        return a_mem
    new_mem = attrs["mem_type"]
    if not new_mem:
        return a_mem
    return new_mem


def concat_memtype_infer(*args, attrs=None):
    if len(args) == 0:
        raise TypeError(
            "Concat op input num should be larger than 0.")
    if not all(args[0] == i for i in args):
        raise TypeError(
            "Concat op's inputs' mem type should be the same, but got {}.".format(" ".join(args)))
    return attrs["mem_type"]


def default_memtype_infer(a_mem, attrs=None):
    pass
