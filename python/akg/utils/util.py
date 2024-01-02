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
from akg.utils.format_transform import get_shape


def parse_int_const(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, (akg.tvm.expr.IntImm, akg.tvm.expr.UIntImm)):
        return value.value
    return None

def parse_workspace_array(workspace):
    total_bytes = []
    workspace_list = get_shape(workspace)
    for ws in workspace_list:
        total_bytes.append(parse_int_const(ws))

    if total_bytes is None:
        return None

    workspace_dict = {
        "num": len(total_bytes),
        "size": total_bytes
    }
    return workspace_dict

def parse_workspace_map(attrs):
    if not isinstance(attrs, akg.tvm.container.Map):
        return None
    if "workspace" not in attrs:
        return None

    workspace_map = attrs["workspace"]
    if not isinstance(workspace_map, akg.tvm.container.Map):
        return None
    if "num" not in workspace_map or "size" not in workspace_map:
        return None

    worksapce_num =  parse_int_const(workspace_map["num"])
    if not isinstance(workspace_map["size"], akg.tvm.container.Array):
        return None

    workspace_size = []
    tt = get_shape(workspace_map["size"])
    for item in tt:
        workspace_size.append(parse_int_const(item))
    if len(workspace_size) != worksapce_num:
        return None

    workspace_dict = {
        "num": worksapce_num,
        "size": workspace_size
    }
    return workspace_dict

def parse_workspace(workspace):
    if not isinstance(workspace, akg.tvm.container.Map):
        return None

    total_bytes = 0
    if "total_bytes" in workspace:
        ws = workspace["total_bytes"]
        if isinstance(ws, akg.tvm.container.Array):
            return parse_workspace_array(ws)
        total_bytes = parse_int_const(ws)

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

def get_ascend_type(desc):
    if "target_info" not in desc.keys():
        return None

    target_info_type = desc["target_info"]
    if target_info_type.get("arch"):
        return target_info_type.get("arch")
    return None