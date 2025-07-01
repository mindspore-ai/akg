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
MMAD_SUPPORT_TYPE = {
    "FP16": {"FP16": "FP32"},
    "INT8": {"INT8": "INT32", "UINT8": "INT32"},
    "UINT8": {"UINT8": "INT32"},
    "INT4": {"INT4": "INT32"}
}

VECTOR_SUPPORT_TYPE = ["FP16", "FP32", "UINT16",
                       "INT16", "UINT32", "INT32", "UINT8", "INT8", "REGIONFP16", "REGIONFP32", "INT4"]
L1_SUPPORT_TYPE = ["FP16", "UINT8", "INT8", "INT4"]
L0C_SUPPORT_TYPE = ["FP32", "INT32"]
GM_SUPPORT_TYPE = ["FP16", "FP32", "UINT16",
                   "INT16", "UINT32", "INT32", "UINT8", "INT8", "INT4"]
ALL_SUPPORT_TYPE = ["BOOL", "INT4", "FP16", "FP32", "UINT16",
                    "INT16", "UINT32", "INT32", "UINT8", "INT8"]

VALID_MEM_TYPE = ["GM", "UB", "L1", "L0A", "L0B", "L0C"]
VALID_FORMAT = ["ND", "NZ", "NCHW", "NHWC", "NC0HWC1", "FM", "FT"]
VALID_DATA_TYPE = {
    "GM": GM_SUPPORT_TYPE,
    "L0A": L1_SUPPORT_TYPE,
    "L0B": L1_SUPPORT_TYPE,
    "L1": L1_SUPPORT_TYPE,
    "L0C": L0C_SUPPORT_TYPE,
    "UB": VECTOR_SUPPORT_TYPE
}
TYPE_SIZE = {
    "BOOL": 1,
    "INT4": 4,
    "FP16": 16,
    "FP32": 32,
    "UINT16": 16,
    "INT16": 16,
    "UINT32": 32,
    "INT32": 32,
    "UINT8": 8,
    "INT8": 8
}

SCALARTYPE_CTYPE = {
    "BOOL": "bool",
    "FP32": "float",
    "INT32": "int",
    "FP16": "half"
}


def is_mem_type_valid(mem_type):
    return mem_type in VALID_MEM_TYPE


def is_dtype_valid(param, mem_type=None):
    if not mem_type:
        return param in ALL_SUPPORT_TYPE
    return param in VALID_DATA_TYPE[mem_type]


def is_format_valid(format):
    return format in VALID_FORMAT


def is_tensor(obj):
    return hasattr(obj, "type") and obj.type == "Tensor"


def is_scalar(obj):
    return hasattr(obj, "type") and obj.type == "Scalar"
