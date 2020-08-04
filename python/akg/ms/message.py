#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019 Huawei Technologies Co., Ltd
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

"""message"""
import importlib.util
import json
import json.decoder as jd
import logging
import traceback
import os.path
from pathlib import Path

import akg.tvm
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg import composite
from akg.tvm import _api_internal
from . import cce
from . import gpu
from . import op_build

@vc_util.check_input_type(str)
def compilewithjson_to_func(json_str):
    """compile with json."""
    try:
        kernel_info = json.loads(json_str)
    except jd.JSONDecodeError:
        logging.error(traceback.format_exc())
        return False

    processor = 'aicore'
    if 'process' in kernel_info:
        processor = kernel_info['process']

    if 'composite' in kernel_info and kernel_info['composite'] is True:
        try:
            if processor == 'cuda':
                _ = composite._build(json_str, kernel_info)
                return True
            else:
                mod = composite._build_to_func(json_str, kernel_info)
                return mod
        except Exception:
            logging.error(traceback.format_exc())
            return False

    op_name = kernel_info['name']
    op_func = None
    # get custom ops implementation first.
    if 'impl_path' in kernel_info and kernel_info['impl_path'] is not None:
        impl_path = os.path.realpath(kernel_info['impl_path'])
        if os.path.isfile(impl_path):
            custom_mod_name = Path(impl_path).resolve().stem
            mod_spec = importlib.util.spec_from_file_location(custom_mod_name, impl_path)
            custom_mod = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(custom_mod)
            op_func = getattr(custom_mod, op_name, None)

    # get built-in ops.
    if op_func is None:
        if processor == 'cuda':
            op_func = getattr(gpu, op_name, None)
            input_shapes = []
            input_types = []
            for input_desc in kernel_info['input_desc']:
                input_shapes.append(input_desc[0]['shape'])
                input_types.append(input_desc[0]['data_type'])
            op_attrs = []
            if kernel_info['attr']:
                for ext_arg in kernel_info['attr']:
                    op_attrs.append(ext_arg['value'])
            dump_ir = os.getenv('MS_AKG_DUMP_IR') == "on"
            dump_code = os.getenv('MS_AKG_DUMP_CODE') == "on"
            mod = utils.op_build(op_func, input_shapes, input_types, op_attrs, kernel_info['op'], dump_ir=dump_ir,
                                 dump_code=dump_code)
            return True
        else:
            op_func = getattr(cce, op_name, None)

    if op_func is None:
        logging.error("this op not support by akg, please check op name %s", str(op_name))
        return False

    args = {}
    tsr = []
    for input_desc in kernel_info['input_desc']:
        if len(input_desc) == 1:
            tensor_shape = input_desc[0]['shape']
            tensor_shape = (1,) if not tensor_shape else tensor_shape
            vc_util.shape_dtype_max_size_check(tensor_shape, input_desc[0]['data_type'])
            args[input_desc[0]['name']] = akg.tvm.placeholder(
                shape=tensor_shape, name=input_desc[0]['tensor_name'], dtype=input_desc[0]['data_type'])
            tsr.append(args[input_desc[0]['name']])
        else:
            tmp_input = []
            for tmp_desc in input_desc:
                tensor_shape = tmp_desc['shape']
                tensor_shape = (1,) if not tensor_shape else tensor_shape
                vc_util.shape_dtype_max_size_check(tensor_shape, tmp_desc['data_type'])
                tmp_input.append(akg.tvm.placeholder(
                    shape=tensor_shape, name=tmp_desc['tensor_name'], dtype=tmp_desc['data_type']))
            args[input_desc[0]['name']] = tmp_input
            tsr = tsr + tmp_input

    if kernel_info['attr']:
        for ext_arg in kernel_info['attr']:
            args[ext_arg['name']] = ext_arg['value']

    output = op_func(**args)
    schedule_func = None
    attrs = {}
    if isinstance(output, (list, tuple)):
        from inspect import isfunction
        tmp_outputs = []
        for elem in output:
            if isfunction(elem):
                schedule_func = elem
            elif isinstance(elem, dict):
                for key, value in elem.items():
                    if key not in attrs or not attrs[key]:
                        attrs[key] = value
            else:
                tmp_outputs.append(elem)

        output = tmp_outputs
    else:
        output = [output]

    tsr = tsr + [i for i in output if utils.TensorUtils.is_output_value(i)]
    return op_build([op_name], output, tsr, schedule_func, processor, kernel_info['op'], attrs)

def compilewithjson(json_str):
    tmp_rst = compilewithjson_to_func(json_str)
    if isinstance(tmp_rst, bool):
        return tmp_rst
    return _api_internal._BuildToModule(tmp_rst)
