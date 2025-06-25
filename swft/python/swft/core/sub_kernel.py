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

from functools import wraps
from inspect import signature
from .c_expression import new_subkernel, new_synckernel

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
