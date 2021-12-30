#!/usr/bin/env python3
# coding: utf-8
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

"""utility"""
import os
import json
import types
import akg.tvm


def parse_int_const(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, (akg.tvm.expr.IntImm, akg.tvm.expr.UIntImm)):
        return value.value
    return None


def parse_workspace(workspace):
    if not isinstance(workspace, akg.tvm.container.Map):
        return None

    total_bytes = 0
    if "total_bytes" in workspace:
        total_bytes = parse_int_const(workspace["total_bytes"])

    if total_bytes is None or total_bytes == 0:
        return None

    workspace_dict = {
        "num": 1,
        "size": [total_bytes]
    }
    return workspace_dict

def parse_kwargs(func, **kwargs):
    if 'target' not in kwargs:
        return kwargs
    if not func or not isinstance(func, types.FunctionType):
        return kwargs
    op_func = func.__dict__["__wrapped__"] if "__wrapped__" in func.__dict__ else func
    args_name = op_func.__code__.co_varnames
    if 'target' not in args_name:
        kwargs.pop('target')
    return kwargs

def write_code(js_dict, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with os.fdopen(os.open(fname, os.O_WRONLY | os.O_CREAT, 0o400), 'w') as f:
        json.dump(js_dict, f, sort_keys=True, indent=4, separators=(',', ':'))

