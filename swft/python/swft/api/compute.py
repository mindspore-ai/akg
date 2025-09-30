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
from .slicedata import concat


@name_tensor
def mmad(src0, src1, src2=None):
    if src2 is None:
        out_mem_type = mmad_memtype_infer(src0.mem_type, src1.mem_type)
        out_size = mmad_shape_infer(src0.shape, src1.shape)
        out_dtype = mmad_dtype_infer(src0.dtype, src1.dtype)
        out_format = mmad_format_infer(src0.format, src1.format)
    else:
        out_mem_type = mmad_memtype_infer(
            src0.mem_type, src1.mem_type, src2.mem_type)
        out_size = mmad_shape_infer(src0.shape, src1.shape, src2.shape)
        out_dtype = mmad_dtype_infer(src0.dtype, src1.dtype, src2.dtype)
        out_format = mmad_format_infer(src0.format, src1.format, src2.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Mmad(src0, src1, out, src2)()
    return out


@name_tensor
def vadd(src0, src1):
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_format = bino_format_infer(src0.format, src1.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vadd(src0, src1, out)()
    return out


@name_tensor
def vsub(src0, src1):
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_format = bino_format_infer(src0.format, src1.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vsub(src0, src1, out)()
    return out


@name_tensor
def vmul(src0, src1):
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_format = bino_format_infer(src0.format, src1.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vmul(src0, src1, out)()
    return out


@name_tensor
def vdiv(src0, src1):
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_format = bino_format_infer(src0.format, src1.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vdiv(src0, src1, out)()
    return out


@name_tensor
def vdivs(src, factor):
    if (not isinstance(factor, Scalar)):
        factor = Scalar(src.dtype, factor)
    one = Scalar(src.dtype, 1)
    factor_reciprocal = one / factor
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = scalar_dtype_infer(src.dtype, factor_reciprocal.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vmuls(src, factor_reciprocal, out)()
    return out


@name_tensor
def vmax(src0, src1):
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_format = bino_format_infer(src0.format, src1.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vmax(src0, src1, out)()
    return out


@name_tensor
def vmin(src0, src1):
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_format = bino_format_infer(src0.format, src1.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vmin(src0, src1, out)()
    return out


@name_tensor
def vand(src0, src1):
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_format = bino_format_infer(src0.format, src1.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vand(src0, src1, out)()
    return out


@name_tensor
def vor(src0, src1):
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_format = bino_format_infer(src0.format, src1.format)
    multi_core = src0.multi_core or src1.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vor(src0, src1, out)()
    return out


@name_tensor
def vadds(src, factor):
    if (not isinstance(factor, Scalar)):
        factor = Scalar(src.dtype, factor)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = scalar_dtype_infer(src.dtype, factor.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vadds(src, factor, out)()
    return out


@name_tensor
def vmuls(src, factor):
    if (not isinstance(factor, Scalar)):
        factor = Scalar(src.dtype, factor)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = scalar_dtype_infer(src.dtype, factor.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vmuls(src, factor, out)()
    return out


@name_tensor
def vsubs(src, factor):
    if (not isinstance(factor, Scalar)):
        factor = Scalar(src.dtype, factor)
    factor = factor * (-1)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = scalar_dtype_infer(src.dtype, factor.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vadds(src, factor, out)()
    return out


@name_tensor
def vmaxs(src, factor):
    if (not isinstance(factor, Scalar)):
        factor = Scalar(src.dtype, factor)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = scalar_dtype_infer(src.dtype, factor.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vmaxs(src, factor, out)()
    return out


@name_tensor
def vmins(src, factor):
    if (not isinstance(factor, Scalar)):
        factor = Scalar(src.dtype, factor)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = scalar_dtype_infer(src.dtype, factor.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vmins(src, factor, out)()
    return out


@name_tensor
def vexp(src):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vexp(src, out)()
    return out


@name_tensor
def vsqrt(src):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vsqrt(src, out)()
    return out


@name_tensor
def vrelu(src):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vrelu(src, out)()
    return out


@name_tensor
def vln(src):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vln(src, out)()
    return out


@name_tensor
def vrec(src):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vrec(src, out)()
    return out


@name_tensor
def vabs(src):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vabs(src, out)()
    return out


@name_tensor
def vnot(src):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vnot(src, out)()
    return out


@name_tensor
def vconv_s42s8(src):
    ub_tmp = vconv_s42f16(src)
    ub_out = vconv(ub_tmp, "INT8")
    return ub_out

@name_tensor
def vconv_s42f16(src):
    # TODO: 当前仅支持离线transpose [4, src.shape]->[src.shape, 4]
    if src.dtype != "INT16":
        raise TypeError("vconv_s42f16 only support type INT16")
    ub_mask_1 = vector_dup(Scalar("INT16", 0xf000), [16], False)
    ub_mask_1_1 = vector_dup(Scalar("INT16", 0x0f00), [16], False)
    ub_mask_1_2 = vector_dup(Scalar("INT16", 0x00f0), [16], False)
    ub_mask_1_3 = vector_dup(Scalar("INT16", 0x000f), [16], False)
    ub_mask_2 = vector_dup(Scalar("INT16", 0x0800), [16], False)
    ub_mask_2_1 = vector_dup(Scalar("INT16", 0x0080), [16], False)
    ub_mask_2_2 = vector_dup(Scalar("INT16", 0x0008), [16], False)
    ub_mask_3 = vector_dup(Scalar("INT16", 0xf000), [16], False)
    ub_mask_3_1 = vector_dup(Scalar("INT16", 0xff00), [16], False)
    ub_mask_3_2 = vector_dup(Scalar("INT16", 0xfff0), [16], False)
    ub_y_0 = vand(src, ub_mask_1)
    ub_y_0_1 = vand(src, ub_mask_1_1)
    ub_y_0_2 = vand(src, ub_mask_1_2)
    ub_y_0_3 = vand(src, ub_mask_1_3)
    ub_y_1 = vsub(ub_mask_2, ub_y_0_1)
    ub_y_1_2 = vsub(ub_mask_2_1, ub_y_0_2)
    ub_y_1_3 = vsub(ub_mask_2_2, ub_y_0_3)
    ub_y_1 = vand(ub_y_1, ub_mask_3)
    ub_y_1_2 = vand(ub_y_1_2, ub_mask_3_1)
    ub_y_1_3 = vand(ub_y_1_3, ub_mask_3_2)
    ub_y_0_1 = vor(ub_y_0_1, ub_y_1)
    ub_y_0_2 = vor(ub_y_0_2, ub_y_1_2)
    ub_y_0_3 = vor(ub_y_0_3, ub_y_1_3)
    ub_y_out_0 = vconv(ub_y_0, "FP16")
    ub_y_out_1 = vconv(ub_y_0_1, "FP16")
    ub_y_out_2 = vconv(ub_y_0_2, "FP16")
    ub_y_out_3 = vconv(ub_y_0_3, "FP16")
    ub_y_out_0 = vmuls(ub_y_out_0, 0.000244140625)
    ub_y_out_1 = vmuls(ub_y_out_1, 0.00390625)
    ub_y_out_2 = vmuls(ub_y_out_2, 0.0625)
    ub_output_list = []
    ub_output_list.append(ub_y_out_0)
    ub_output_list.append(ub_y_out_1)
    ub_output_list.append(ub_y_out_2)
    ub_output_list.append(ub_y_out_3)
    ub_out = concat(ub_output_list, 0)
    return ub_out


@name_tensor
def vconv(src, dtype, type=None):
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = conv_dtype_infer(src.dtype, {"dtype": dtype})
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    if type is None:
        attrs = {"dtype": dtype}
    else:
        attrs = {"dtype": dtype, type: [1.0]}
    Vconv(src, out, attrs)()
    return out


@name_tensor
def vcmax(src, reduce_axis):
    if (reduce_axis < 0):
        reduce_axis += len(src.shape)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = reduce_shape_infer(
        src.shape, attrs={"reduce_axis": [reduce_axis]})
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    in_shape = [float(x.value) for x in src.shape]
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vcmax(src, out, attrs={"reduce_axis": [reduce_axis], "pad_colum":in_shape})()
    return out


@name_tensor
def vcmin(src, reduce_axis):
    if (reduce_axis < 0):
        reduce_axis += len(src.shape)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = reduce_shape_infer(
        src.shape, attrs={"reduce_axis": [reduce_axis]})
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    in_shape = [float(x.value) for x in src.shape]
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vcmin(src, out, attrs={"reduce_axis": [reduce_axis], "pad_colum":in_shape})()
    return out


@name_tensor
def vcadd(src, reduce_axis):
    if (reduce_axis < 0):
        reduce_axis += len(src.shape)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = reduce_shape_infer(
        src.shape, attrs={"reduce_axis": [reduce_axis]})
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    in_shape = [float(x.value) for x in src.shape]
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vcadd(src, out, attrs={"reduce_axis": [reduce_axis], "pad_colum":in_shape})()
    return out


@name_tensor
def vbrcb(src, broadcast_axis, broad_size):
    if (broadcast_axis < 0):
        broadcast_axis += len(src.shape)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = brcb_shape_infer(
        src.shape, attrs={"broadcast_axis": [broadcast_axis], "broad_size": [broad_size]})
    out_dtype = mono_dtype_infer(src.dtype)
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vbrcb(src, out, attrs={"broadcast_axis": [
          broadcast_axis], "broad_size": [broad_size]})()
    return out


@name_tensor
def vector_dup(factor, size, multi_core, no_autopad=False):
    if (not isinstance(factor, Scalar)):
        raise TypeError("VectorDup requires Scalar input.")
    out_mem_type = dup_memtype_infer(None)
    out_size = dup_shape_infer(None, {"size": size})
    out_dtype = mono_dtype_infer(factor.dtype)
    out_format = dup_format_infer(None)
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    attrs = {"size": size}
    if no_autopad:
        attrs["no_pad"] = [1.0]
    VectorDup(factor, out, attrs)()
    return out


@name_tensor
def vcmpv(src0, src1, opType):
    if opType not in CMPV_SUPPORT_OPTYPE:
        raise TypeError("vcmpv not support optype {}".format(opType))
    out_dtype = cmpv_dtype_infer(src0.dtype, src1.dtype)
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = bino_shape_infer(src0.shape, src1.shape)
    out_format = default_format_infer(src0.format)
    multi_core = src0.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vcmpv(src0, src1, out, attrs={"opType": [CMP_OP_TYPE[opType]]})()
    return out


@name_tensor
def vcmpvs(src, factor, opType):
    if (not isinstance(factor, Scalar)):
        factor = Scalar(src.dtype, factor)
    out_mem_type = mono_memtype_infer(src.mem_type)
    out_size = default_shape_infer(src.shape)
    out_dtype = "BOOL"
    out_format = default_format_infer(src.format)
    multi_core = src.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vcmpvs(src, factor, out, attrs={"opType": [CMP_OP_TYPE[opType]]})()
    return out


@name_tensor
def where(src0, src1, condition):
    out_dtype = bino_dtype_infer(src0.dtype, src1.dtype)
    out_mem_type = bino_memtype_infer(src0.mem_type, src1.mem_type)
    out_size = condition.shape
    out_format = default_format_infer(src0.format)
    multi_core = src0.multi_core
    out = Tensor(out_mem_type, out_dtype, out_size, out_format, multi_core)
    Vsel(src0, src1, condition, out)()
    return out
