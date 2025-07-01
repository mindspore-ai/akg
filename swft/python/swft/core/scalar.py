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

from .c_expression import Scalar as CScalar
from .instruction import Instruction
from .name_tensor import name_tensor
import swft.utils as checker
import traceback
from functools import reduce


class Scalar(CScalar):
    def __init__(self, dtype, value=None):
        if dtype not in ["INT32", "FP32", "BOOL", "FP16", "INT16"]:
            raise TypeError(
                "Currently only support Scalar with INT32, FP32, FP16, INT16 or BOOL")
        if (value is not None):
            CScalar.__init__(self, dtype, value)
        else:
            CScalar.__init__(self, dtype)
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        parse = code.split("=")
        if len(parse) > 1:
            self.__name__ = parse[0].strip()

    @property
    def type(self):
        return "Scalar"

    @property
    def value(self):
        if not self.has_value():
            return ValueError("Scalar value not defined!")
        return self.getValue()

    @property
    def dtype(self):
        return self.getDtype()

    @name_tensor
    def astype(self, dtype):
        if not checker.is_dtype_valid(dtype):
            raise TypeError("Scalar astype dtype not valid")
        if self.has_value():
            return Scalar(dtype, self.value)
        res = Scalar(dtype)
        Instruction("SCAST", (self, ), (res, ), None)()
        return res

    def load(self, scalar, attr=None):
        if isinstance(scalar, (int, float, bool)):
            scalar = Scalar(self.dtype, scalar)
        if hasattr(scalar, 'shape'):
            allsize = reduce(lambda x, y: x*y, scalar.shape)
        if hasattr(scalar, 'shape') and allsize != 1:
            raise ValueError(
                "For Scalar load(Tensor), Tensor shape must be [1]")
        Instruction("MOV", (scalar, ), (self, ), attr)()

    @name_tensor
    def copy(self):
        adder = Scalar(self.dtype, 0)
        res = Scalar(self.dtype)
        Instruction("SADD", (self, adder, ), (res, ), None)()
        return res

    @name_tensor
    def sqrt(self):
        if self.has_value():
            return Scalar(self.dtype, self.value ** 0.5)
        res = Scalar(self.dtype)
        Instruction("SSQRT", (self, ), (res, ), None)()
        return res

    @name_tensor
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar add Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, self.value + other.value)
        res = Scalar(self.dtype)
        Instruction("SADD", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __radd__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar add Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, self.value + other.value)
        res = Scalar(self.dtype)
        Instruction("SADD", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar sub Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, self.value - other.value)
        res = Scalar(self.dtype)
        Instruction("SSUB", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar sub Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value - self.value)
        res = Scalar(self.dtype)
        Instruction("SSUB", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar mul Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, self.value * other.value)
        res = Scalar(self.dtype)
        Instruction("SMUL", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar mul Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value * self.value)
        res = Scalar(self.dtype)
        Instruction("SMUL", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar div Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, self.value / other.value)
        res = Scalar(self.dtype)
        Instruction("SDIV", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar div Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value / self.value)
        res = Scalar(self.dtype)
        Instruction("SDIV", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError(
                "Currently only support Scalar floordiv Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, self.value // other.value)
        res = Scalar(self.dtype)
        Instruction("SFLOORDIV", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rfloordiv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError(
                "Currently only support Scalar floordiv Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value // self.value)
        res = Scalar(self.dtype)
        Instruction("SFLOORDIV", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __mod__(self, other):
        if isinstance(other, int):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar mod Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, self.value % other.value)
        res = Scalar(self.dtype)
        Instruction("SMOD", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rmod__(self, other):
        if isinstance(other, int):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar mod Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar(self.dtype)
        Instruction("SMOD", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __rmod__(self, other):
        if isinstance(other, int):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar mod Scalar/int")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar(self.dtype)
        Instruction("SMOD", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar < Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SLESSTHAN", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rlt__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar < Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SLESSTHAN", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar > Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SGREATERTHAN", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rgt__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError("Currently only support Scalar > Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SGREATERTHAN", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __le__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError(
                "Currently only support Scalar <= Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SLESSEQUAL", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rle__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError(
                "Currently only support Scalar <= Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SLESSEQUAL", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __ge__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError(
                "Currently only support Scalar >= Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SGREATEREQUAL", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rge__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError(
                "Currently only support Scalar >= Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SGREATEREQUAL", (other, self, ), (res, ), None)()
        return res

    @name_tensor
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError(
                "Currently only support Scalar == Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SEQUAL", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __req__(self, other):
        return self.__eq__(other)

    @name_tensor
    def __ne__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(self.dtype, other)
        elif not isinstance(other, Scalar):
            raise TypeError(
                "Currently only support Scalar != Scalar/int/float")
        if self.has_value() and other.has_value():
            return Scalar(self.dtype, other.value % self.value)
        res = Scalar("BOOL")
        Instruction("SNOTEQUAL", (self, other, ), (res, ), None)()
        return res

    @name_tensor
    def __rne__(self, other):
        return self.__ne__(other)

    @name_tensor
    def __neg__(self):
        if self.has_value():
            return Scalar(self.dtype, -self.value)
        res = Scalar(self.dtype)
        Instruction("SNEG", (self, ), (res, ), None)()
