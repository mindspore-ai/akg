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

from .c_expression import push_to_list, compile_ckernel
from swft.utils import is_scalar
trace = []


def trace_info(idx, name, inputs, outputs, attr):
    list_keys = []
    list_values = []
    if attr is not None:
        for (k, v) in attr.items():
            if (isinstance(v, list)):
                list_keys.append(k)
                list_values.append([s.value if is_scalar(s) else s for s in v])
    input_tensors = []
    input_scalars = []
    output_tensors = []
    output_scalars = []
    for inp in inputs:
        if (inp.type == "Tensor"):
            input_tensors.append(inp)
        elif (inp.type == "Scalar"):
            input_scalars.append(inp)
    for out in outputs:
        if (out.type == "Tensor"):
            output_tensors.append(out)
        elif (out.type == "Scalar"):
            output_scalars.append(out)
    push_to_list(idx, name, input_tensors, input_scalars, output_tensors, output_scalars,
                 list_keys, list_values)


def add_trace(idx, name, inputs, outputs, attr):
    global trace
    trace.append((idx, name, inputs, outputs, attr))


def compile_kernel(file_path, name=None, hard_sync=False, idx=-1):
    global trace
    for i in trace:
        trace_info(*i)
    trace.clear()
    if name == None:
        succ = compile_ckernel(file_path, "", hard_sync, idx)
        if not succ:
            raise Exception("Compile failed, not good tiling size.")
    else:
        succ = compile_ckernel(file_path, name, hard_sync, idx)
        if not succ:
            raise Exception("Compile failed, not good tiling size.")
