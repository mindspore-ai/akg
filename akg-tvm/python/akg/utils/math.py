#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019 Huawei Technologies Co., Ltd
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

"""math function"""
def greatest_common_divisor(x, y):
    """get the greatest common divisor of rhs, lhs."""
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("Input of greatest common divisor should be integer")
    if y < x:
        y, x = x, y
    if x == 0:
        raise ValueError("Input can not be zero")
    z = x

    while y % x != 0:
        z = y % z
        y = x
        x = z
    return z


def least_common_multiple(x, y):
    """get the least common multiple of rhs, lhs."""
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("Input of least common multiple should be integer")
    return x * y / greatest_common_divisor(x, y)
