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

from .c_expression import Tensor as CTensor
from .instruction import Instruction
from .scalar import Scalar
from .name_tensor import name_tensor

from swft.utils import util as checker
import traceback
from copy import deepcopy


class Tensor(CTensor):
    def __init__(self, mem_type, dtype, shape, format, multi_core=None):
        CTensor.__init__(self, mem_type, dtype, shape, format, multi_core)
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        parse = code.split("=")
        if len(parse) > 1:
            self.__name__ = parse[0].strip()

    @property
    def type(self):
        return "Tensor"

    @property
    def shape(self):
        return self.getShape()

    @property
    def dtype(self):
        return self.getDtype()

    @property
    def mem_type(self):
        return self.getMemType()

    @property
    def format(self):
        return self.getFormat()

    @property
    def multi_core(self):
        return self.getMultiCore()

    @name_tensor
    def load(self, tensor, attr=None):
        if not isinstance(tensor, Tensor):
            raise TypeError(
                "For Tensor.load, input should be Tensor, but got {}".format(type(tensor)))

        if tensor.dtype != self.dtype and tensor.mem_type != "L0C" and self.mem_type != "L0C":
            raise TypeError("dtype({} and {}) not match".format(
                self.dtype, tensor.dtype))

        if self.mem_type != tensor.mem_type and self.mem_type != "L0C":
            support_shape = deepcopy(tensor.shape)
            if attr is not None:
                support_shape = deepcopy(self.shape)
            if self.shape != support_shape:
                raise Exception("Tensor shape {} and {} not match.".format(
                    support_shape, self.shape))
            if self.format != tensor.format:
                raise Exception("Tensor format {} and {} not match.".format(
                    tensor.format, self.format))
        valid_path = {"L0C": ["UB"], "UB": ["GM", "L0C", "UB", "L1"], "L1": ["UB", "GM"], "L0A": ["L1", "GM"],
                      "L0B": ["L1", "GM"], "GM": ["GM", "UB"]}
        if tensor.mem_type not in valid_path[self.mem_type]:
            raise ValueError("Cannot move Tensor from {} to {}.".format(
                tensor.mem_type, self.mem_type))

        if self.multi_core is None:
            self.multi_core = tensor.multi_core
        self.update_param(
            self.dtype, self.shape, self.format, self.multi_core)
        Instruction("MOV", (tensor, ), (self, ), attr)()

    @name_tensor
    def __getitem__(self, index):
        if self.mem_type not in ("GM", "UB", "L1"):
            raise TypeError("Tensor __getitem__ only supports GM/UB/L1")
        instr_str = "SLICE" if self.mem_type == "GM" else "SLICEMOV"
        slice_shape = deepcopy(self.shape)
        begin = [0] * len(slice_shape)
        if not isinstance(index, tuple):
            index = (index,)
        for i in range(len(self.shape)):
            if i >= len(index):
                continue
            if isinstance(index[i], (int, Scalar)):
                begin[i] = index[i]
                slice_shape[i] = 1
            elif isinstance(index[i], slice):
                if index[i].step and index[i].step != 1:
                    raise TypeError(
                        "Tensor __getitem__ only supports slice with stride=1")
                begin_i = index[i].start if (index[i].start is not None) else 0
                end_i = index[i].stop if (
                    index[i].stop is not None) else self.shape[i]
                shape_i = end_i - begin_i
                if isinstance(shape_i, Scalar):
                    if not shape_i.value:
                        raise ValueError(
                            "Tensor __getitem__ slice shape cannot be inferred!")
                    if not isinstance(shape_i.value, int):
                        raise TypeError(
                            "Tensor __getitem__ slice shape must be int!")
                    if shape_i.value <= 0:
                        raise ValueError(
                            "Tensor __getitem__ slice shape should be greated than 0!")
                    slice_shape[i] = shape_i.value
                else:
                    if not isinstance(shape_i, int):
                        raise TypeError(
                            "Tensor __getitem__ slice shape format error!")
                    slice_shape[i] = shape_i
            else:
                raise TypeError(
                    "Tensor __getitem__ only support int or slice!")
        dst = Tensor(self.mem_type, self.dtype, list(
            slice_shape), self.format, self.multi_core)
        slice_shape = (Scalar("INT32", x) for x in slice_shape)
        begin = (Scalar("INT32", x) if isinstance(
            x, int) else x for x in begin)
        Instruction(instr_str, (self,) + tuple(begin) +
                    tuple(slice_shape), (dst, ), None)()
        return dst

    def __setitem__(self, index, item):
        raise NotImplementedError("Tensor __setitem__ currently not supported")
