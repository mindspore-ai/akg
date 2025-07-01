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
import traceback
import keyword
import re
from swft.utils.util import is_scalar, is_tensor


def get_second_argument(function_call):
    match = re.search(r'\((.*)\)', function_call)
    if not match:
        return None
    args = [arg.strip() for arg in match.group(1).split(',')]
    return args[1] if len(args) > 1 else None


def is_valid_variable_name(name):
    if keyword.iskeyword(name):
        return False
    if not name.isidentifier():
        return False
    return True


def name_tensor(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        parse = code.split("=")
        res = f(*args, **kwargs)
        if len(parse) > 1:
            if re.search("move_scalar_to_ub", parse[1]) and (not re.search(get_second_argument(parse[1]), parse[0])):
                raise TypeError(
                    "for move_scalar_to_ub, input and output must be the same ub address")
            tensor_name = parse[0].strip()
            if is_valid_variable_name(tensor_name):
                if is_tensor(res) or is_scalar(res):
                    if res.get_name() == "":
                        res.update_name(tensor_name)
                elif isinstance(res, list):
                    for i in res:
                        if i.get_name() == "":
                            i.update_name(tensor_name)
        return res
    return decorated
