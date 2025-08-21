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

import os
import sys
import importlib
import importlib.util
from functools import wraps, lru_cache
from inspect import signature
from .c_expression import new_subkernel, new_synckernel
from .compile import compile_kernel
from swft.runtime.npu_session import NPUSession
from swft.utils import is_tensor, is_scalar, gen_opbind_cpp
kernel_list = []


def sub_kernel(core_num=1):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            global kernel_list
            kernel_id = len(kernel_list)
            var_names = list(signature(func).parameters.keys())
            pos = 0
            for idx in range(len(var_names)):
                if isinstance(args[idx], int) or isinstance(args[idx], bool):
                    continue
                args[idx].update_name(args[idx].__name__)
                args[idx].update_position(kernel_id, pos)
                pos += 1
            new_subkernel(core_num, func.__name__)
            kernel_list.append(kernel_id)
            return func(*args, **kwargs)
        return wrapped_function
    return logging_decorator


@lru_cache(maxsize=None)
def load_func_from_module(module_name, func_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def call_op(kernel_name, so_path, core_num, args):
    c_args = [core_num]
    for arg in args:
        if isinstance(arg, int) or isinstance(arg, bool):
            c_args.append(arg)
            continue
        if is_tensor(arg):
            c_args.append(arg.device_data_ptr())
        if is_scalar(arg):
            c_args.append(arg.value)
    c_args.append(NPUSession.instance().stream)
    load_func_from_module(kernel_name, kernel_name, so_path)(*c_args)


def native_jit(core_num=1):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            prefix_path = "temp"
            kernel_name = func.__name__
            so_file = f"./{prefix_path}/{kernel_name}/{kernel_name}.so"
            if os.path.exists(so_file):
                call_op(kernel_name, so_file, core_num, args)
                return wrapped_function
            os.system(f"mkdir -p {prefix_path}/{kernel_name}")
            npu_session = NPUSession.instance()
            global kernel_list
            kernel_id = len(kernel_list)
            var_names = list(signature(func).parameters.keys())
            pos = 0
            c_args = [core_num]
            for idx in range(len(var_names)):
                if isinstance(args[idx], int) or isinstance(args[idx], bool):
                    c_args.append(args[idx])
                    continue
                if is_tensor(args[idx]):
                    c_args.append(args[idx].device_data_ptr())
                if is_scalar(args[idx]):
                    c_args.append(args[idx].value)
                args[idx].update_name(args[idx].__name__)
                args[idx].update_position(kernel_id, pos)
                pos += 1
            c_args.append(npu_session.stream)
            new_subkernel(core_num, kernel_name)
            kernel_list.append(kernel_id)
            func(*args, **kwargs)
            compile_kernel(
                f"./{prefix_path}/{kernel_name}/{kernel_name}.cce", kernel_name, idx=kernel_id)
            cann_path = npu_session.cann_path
            compile_opt = f'{cann_path}/toolkit/tools/ccec_compiler/bin/ccec -xcce --cce-aicore-arch=dav-m200 -mllvm \
                         -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false -mllvm \
                         -cce-aicore-addr-transform -mllvm --cce-aicore-jump-expand=true -fPIC -pthread -o \
                         {prefix_path}/{kernel_name}/{kernel_name}.o -c {prefix_path}/{kernel_name}/{kernel_name}.cce'
            with open(f'./{prefix_path}/{kernel_name}/pybind.cpp', 'w') as f:
                f.write("\n".join(gen_opbind_cpp(
                    f"./{prefix_path}/{kernel_name}/{kernel_name}.cce", kernel_name, args)))
            link_opt = f"{cann_path}/toolkit/tools/ccec_compiler/bin/ccec --cce-fatobj-link -O3 -Wall -shared \
                         -std=c++11 -fPIC `python -m pybind11 --includes` ./{prefix_path}/{kernel_name}/pybind.cpp \
                         -o ./{prefix_path}/{kernel_name}/{kernel_name}.so ./{prefix_path}/{kernel_name}/{kernel_name}.o -I {cann_path}/acllib/include \
                         -L {cann_path}/lib64/ -L {cann_path}/aarch64-linux/lib64/ -Wl,-Bdynamic -lstdc++ -lruntime \
                         -lprofapi -lascendcl"
            os.system(compile_opt)
            os.system(link_opt)
            spec = importlib.util.spec_from_file_location(kernel_name, so_file)
            op_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(op_module)
            function = getattr(op_module, kernel_name)
            function(*c_args)
        return wrapped_function
    return logging_decorator


def sync_kernel():
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            global kernel_list
            kernel_id = len(kernel_list)
            var_names = list(signature(func).parameters.keys())
            for idx in range(len(var_names)):
                args[idx].update_name(args[idx].__name__)
                args[idx].update_position(kernel_id, idx)
            new_synckernel(1, func.__name__, args[0])
            kernel_list.append(kernel_id)
            return func(*args, **kwargs)
        return wrapped_function
    return logging_decorator


def get_idx():
    global kernel_list
    return len(kernel_list) - 1
