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

class Checker():
    def __init__(self, name, memtype_infer, dtype_infer, format_infer, shape_infer):
        self.name = name
        self.memtype_infer = memtype_infer
        self.dtype_infer = dtype_infer
        self.format_infer = format_infer
        self.shape_infer = shape_infer

    def __call__(self, inputs, outputs, attrs=None):
        mems = []
        types = []
        formats = []
        shapes = []
        for x in inputs:
            if x.type == "Tensor":
                mems.append(x.mem_type)
                formats.append(x.format)
                shapes.append(x.shape)
            types.append(x.dtype)
        if self.memtype_infer is not None:
            out_mem = self.memtype_infer(*mems, attrs=attrs)
            if out_mem != outputs[0].mem_type:
                raise TypeError("{} mem_type mismatch, should be {}, but got {}.".format(
                    self.name, out_mem, outputs[0].mem_type))
        if self.dtype_infer is not None:
            out_type = self.dtype_infer(*types, attrs=attrs)
            if (out_type != outputs[0].dtype):
                raise TypeError("{} type mismatch, should be {}, but got {}.".format(
                    self.name, out_type, outputs[0].dtype))
        if self.format_infer is not None:
            out_format = self.format_infer(*formats, attrs=attrs)
            if (out_format != outputs[0].format):
                raise TypeError("{} format mismatch, should be {}, but got {}.".format(
                    self.name, out_format, outputs[0].format))
        if self.shape_infer is not None:
            out_shape = self.shape_infer(*shapes, attrs=attrs)
            if (out_shape != outputs[0].shape):
                raise TypeError("{} shape mismatch, should be {}, but got {}.".format(
                    self.name, out_shape, outputs[0].shape))
