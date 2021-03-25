#!/usr/bin/env python3
# coding: utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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

"""composite topi"""
from akg import tvm

@tvm.register_func("ElemAny")
def elem_any(inputs, attrs):
    def kernel_ir(dst, data):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(data.shape, "ax") as i:
            zero = tvm.const(0, data.dtype)
            one = tvm.const(1, data.dtype)
            with ib.if_scope(ib.load(data, i) > zero):
                ib.store(dst, 0, one)
        return ib.get()
    in_tensor = inputs[0]
    return tvm.extern((1,), [in_tensor], lambda ins, outs : kernel_ir(outs[0], ins[0]), name = "elemany", dtype=in_tensor.dtype)

@tvm.register_func("ElemAll")
def elem_all(inputs, attrs):
    def kernel_ir(dst, data):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(data.shape, "ax") as i:
            zero = tvm.const(0, data.dtype)
            with ib.if_scope(ib.load(data, i) == zero):
                ib.store(dst, 0, zero)
        return ib.get()
    in_tensor = inputs[0]
    return tvm.extern((1,), [in_tensor], lambda ins, outs : kernel_ir(outs[0], ins[0]), name = "elemall", dtype=in_tensor.dtype)

