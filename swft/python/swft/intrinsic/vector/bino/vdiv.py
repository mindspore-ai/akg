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

from swft.core import Instruction
from swft.utils import *


class Vdiv(Instruction):
    def __init__(self, x, y, out):
        self.inputs = (x, y)
        self.outputs = (out,)
        self.checker = Checker(
            "VDIV", bino_memtype_infer, bino_dtype_infer, bino_format_infer, bino_shape_infer)
        super(Vdiv, self).__init__("VDIV", self.inputs, self.outputs)

    def instr_check(self, inputs, outputs, attr):
        self.checker(inputs, outputs, attr)
