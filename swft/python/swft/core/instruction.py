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

from .compile import add_trace
from .sub_kernel import get_idx


class Instruction():
    def __init__(self, name, inputs, outputs, attr=None):
        self.name = name
        self.outputs = outputs
        self.inputs = inputs
        self.attr = attr

    def __call__(self, *args, **kwargs):
        self.instr_check(self.inputs, self.outputs, self.attr)
        add_trace(get_idx(), self.name, self.inputs, self.outputs, self.attr)

    def instr_check(self, inputs, outputs, attr):
        pass
