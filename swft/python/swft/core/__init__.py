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

from .tensor import Tensor
from .scalar import Scalar
from .c_expression import set_context, get_context
from .dynamic_loop import dynamic_loop
from .sub_kernel import sub_kernel, sync_kernel, native_jit
from .instruction import Instruction
from .compile import compile_kernel
from .compile_func import compile_func, code_block_context, custom_and, custom_or, custom_not
from .name_tensor import name_tensor
import os
if os.getenv("ENABLE_SWFT_JIT", 1):
  from .ms_plugin import jit, aot, compile_ms_cell, compile_ms_func
