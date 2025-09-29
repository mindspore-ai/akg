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

from swft.core import Tensor, Scalar, name_tensor
from swft.intrinsic import *
from copy import deepcopy
from swft.utils import *


@name_tensor
def vconcat(src0, src1=None):
    shape = deepcopy(src0.shape)
    shape[-1] = shape[-1] * 8
    dst = Tensor("UB", src0.dtype, shape, src0.format, src0.multi_core)
    if (src1 is None):
        Instruction("VCONCAT", (src0, ), (dst, ), {"type": [4]})()
        return dst
    Instruction("VCONCAT", (src0, src1, ), (dst, ), {"type": [4, 5]})()
    return dst


@name_tensor
def vsort16(src0):
    dst = Tensor("UB", src0.dtype, src0.shape, src0.format, src0.multi_core)
    Instruction("VBITSORT16", (src0, ), (dst, ))()
    return dst


@name_tensor
def vmrgsort4(src0, src1=None, src2=None, src3=None, len0=None, len1=None, len2=None, len3=None, rep=1, all_local=True):
    shape = deepcopy(src0.shape)
    if (len0 is None):
        len0_s = Scalar("INT32", shape[-1])
    elif (isinstance(len0, int)):
        len0_s = Scalar("INT32", len0 * 8)
        shape[-1] = len0 * 8 * rep
    else:
        len0_s = len0 * 8
    is_exhasuted = 0.0 if all_local else 1.0
    if (src1 is None):
        dst = Tensor("UB", src0.dtype, shape, src0.format, src0.multi_core)
        Instruction("VMRGSORT4", (src0, len0_s), (dst, ), {"config": [
                    1], "rep": [rep], "is_exhasuted": [is_exhasuted]})()
        return dst

    if (len1 is None):
        len1_s = Scalar("INT32", src1.shape[-1])
        shape[-1] += src1.shape[-1]
    elif (isinstance(len1, int)):
        len1_s = Scalar("INT32", len1 * 8)
        shape[-1] += len1 * 8 * rep
    else:
        len1_s = len1 * 8
        shape[-1] += src1.shape[-1]
    if (src2 == None):
        dst = Tensor("UB", src0.dtype, shape, src0.format, src0.multi_core)
        Instruction("VMRGSORT4", (src0, len0_s, src1, len1_s), (dst, ), {
                    "config": [3], "rep": [rep], "is_exhasuted": [is_exhasuted]})()
        return dst

    if (len2 is None):
        len2_s = Scalar("INT32", src2.shape[-1])
        shape[-1] += src2.shape[-1]
    elif (isinstance(len2, int)):
        len2_s = Scalar("INT32", len2 * 8)
        shape[-1] += len2 * 8 * rep
    else:
        len2_s = len2 * 8
        shape[-1] += src2.shape[-1]
    if (src3 is None):
        dst = Tensor("UB", src0.dtype, shape, src0.format, src0.multi_core)
        Instruction("VMRGSORT4", (src0, len0_s, src1, len1_s, src2, len2_s),
                    (dst, ), {"config": [7], "rep": [rep], "is_exhasuted": [is_exhasuted]})()
        return dst
    
    if (len3 is None):
        len3_s = Scalar("INT32", src3.shape[-1])
        shape[-1] += src3.shape[-1]
    elif (isinstance(len3, int)):
        len3_s = Scalar("INT32", len3 * 8)
        shape[-1] += len3 * 8 * rep
    else:
        len3_s = len3 * 8
        shape[-1] += src3.shape[-1]
    dst = Tensor("UB", src0.dtype, shape, src0.format, src0.multi_core)
    Instruction("VMRGSORT4", (src0, len0_s, src1, len1_s, src2, len2_s, src3, len3_s),
                (dst, ), {"config": [15], "rep": [rep], "is_exhasuted": [is_exhasuted]})()
    return dst

@name_tensor
def vextract(src):
    shape = deepcopy(src.shape)
    shape[-1] = shape[-1] // 8
    dst0 = Tensor("UB", src.dtype, shape, src.format, src.multi_core)
    Instruction("VEXTRACT", (src, ), (dst0, ), {"type": [4]})()
    dst1 = Tensor("UB", src.dtype, shape, src.format, src.multi_core)
    Instruction("VEXTRACT", (src, ), (dst1, ), {"type": [5]})()
    return dst0, dst1

def mgr_sort(x_in, index_in, x_out, index_out, len, tiling):
    if (isinstance(len, int)):
        len = Scalar("INT32", len)
    if (isinstance(tiling, int)):
        tiling = Scalar("INT32", tiling)
    Instruction("MGRSORT", (x_in, index_in, x_out, index_out, len, tiling), (x_in, index_in, x_out, index_out))()