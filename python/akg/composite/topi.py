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
import akg.topi as topi
from akg import tvm
from akg.utils.format_transform import get_const
from akg.utils import validation_check as vc_util


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
    return tvm.extern((1,), [in_tensor], lambda ins, outs : kernel_ir(outs[0], ins[0]),
                      name = "elemany", dtype=in_tensor.dtype)

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
    return tvm.extern((1,), [in_tensor], lambda ins, outs : kernel_ir(outs[0], ins[0]),
                      name = "elemall", dtype=in_tensor.dtype)

@tvm.register_func("PadAkg")
def pad(inputs, attrs):
    if len(inputs) != 1:
        raise ValueError("Num of inputs should be 1, but got %d." % len(inputs))
    in_tensor = inputs[0]
    attrs = {k: v for k, v in attrs.items()}
    pad_before = attrs["head"]
    pad_after = attrs["tail"]
    pad_value = attrs["pad_val"]
    n = len(in_tensor.shape)
    if len(pad_before) != n or len(pad_after) != n:
        raise ValueError(
            "Input dimensions and pad dimensions dismatch: %d vs %d vs %d" % (n, len(pad_before), len(pad_after)))
    output_name = "T_pad_" + in_tensor.op.name
    return topi.nn.pad(in_tensor, pad_before, pad_after, pad_value, name=output_name)

@tvm.register_func("UnPadAkg")
def unpad(inputs, attrs):
    def kernel_ir(dst, data):
        ib = tvm.ir_builder.create()
        original_shape_ = [get_const(x) for x in data.shape]
        m0, n0 = original_shape_[-2:]
        unpad_shape_ = [get_const(x) for x in unpad_after]
        m1, n1 = unpad_shape_[-2:]
        batch_dims = data.shape[:-2]

        with ib.for_range_n(batch_dims, "bs") as i:
            with ib.for_range(0, m0 - m1) as i_m1:
                with ib.for_range(0, n0 - n1) as i_n1:
                    output_args = i + [i_m1, i_n1]
                    input_args = i + [i_m1, i_n1]
                    ib.store(dst, output_args, ib.load(data, input_args))
        return ib.get()

    if len(inputs) != 1:
        raise ValueError("Num of inputs should be 1, but got %d." % len(inputs))

    in_tensor = inputs[0]
    attrs = {k: v for k, v in attrs.items()}
    n = len(in_tensor.shape)
    unpad_after = attrs["tail"]
    if n < 2:
        raise ValueError("dimensions of input should greater than 1, but got %d." % n)
    if len(unpad_after) != n:
        raise ValueError("Input dimensions and unpad dimensions dismatch: %d vs %d" % (n, len(unpad_after)))
    output_shape = [in_tensor.shape[i] - unpad_after[i] for i in range(0, n)]
    output_name = "T_unpad_" + in_tensor.op.name
    return tvm.extern(output_shape, [in_tensor], lambda ins, outs: kernel_ir(outs[0], ins[0]),
        name = output_name, dtype=[in_tensor.dtype])

@tvm.register_func("CReal")
def creal(inputs, attrs):
    in_tensor = inputs[0]
    out_shape = in_tensor.shape[:-1]
    def fcompute(*index):
        out_index = [x for x in index]
        out_index.append(0)
        return in_tensor(*out_index)
    return tvm.compute(out_shape, fcompute, name = "real")

@tvm.register_func("CImag")
def cimag(inputs, attrs):
    in_tensor = inputs[0]
    out_shape = in_tensor.shape[:-1]
    def fcompute(*index):
        out_index = [x for x in index]
        out_index.append(1)
        return in_tensor(*out_index)
    return tvm.compute(out_shape, fcompute, name = "imag")

@tvm.register_func("Complex")
def complex(inputs, attrs):
    def mix_func(dst, real, imag):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(real.shape, "i") as i:
            ib.store(dst, i + [0], ib.load(real, i))
            ib.store(dst, i + [1], ib.load(imag, i))
        return ib.get()
    real, imag = inputs[0], inputs[1]
    shape = [x for x in real.shape]
    shape.append(2)
    return tvm.extern(shape, [real, imag],
                      lambda ins, outs : mix_func(outs[0], ins[0], ins[1]),
                      name = "complex", dtype=real.dtype)

@tvm.register_func("TransData")
def trans_data(inputs, attrs):
    attrs = {k: v for k, v in attrs.items()}
    if len(inputs) != 1:
        raise ValueError("length of inputs shoule be 1, but got %d." % len(inputs))
    if "src_format" not in attrs or "dst_format" not in attrs:
        raise ValueError("src_format or dst_format not be found in the attrs")
    input_data = inputs[0]
    output_name = "T_transdata_" + input_data.op.name
    src_format = attrs["src_format"]
    dst_format = attrs["dst_format"]
    input_dtype = input_data.dtype
    vc_util.ops_dtype_check(input_dtype,
                            [vc_util.DtypeForDavinci.FLOAT16, vc_util.DtypeForDavinci.FLOAT32])
    # cube size is 16
    cs = 16

    def _zn2default(data, original_shape):
        if len(data.shape) < 4:
            raise ValueError("length of shape of input_data should be greater than or equal to 4, but got %d"
                             % len(data.shape))
        if len(original_shape) < 2:
            raise ValueError("length of original_shape(output_shape) should be greater than or equal to 2, but got %d"
                             % len(original_shape))

        def kernel_ir(input_, output):
            ib = tvm.ir_builder.create()
            shape = [get_const(x) for x in input_.shape]
            n1, m1, m0, n0 = shape[-4:]
            original_shape_ = [get_const(x) for x in original_shape]
            m, n = original_shape_[-2:]
            batch_dims = shape[:-4]

            with ib.for_range_n(batch_dims, "bs") as i:
                with ib.for_range(0, n1) as i_n1:
                    with ib.for_range(0, m1) as i_m1:
                        with ib.for_range(0, m0) as i_m0:
                            with ib.for_range(0, n0) as i_n0:
                                with ib.if_scope(tvm.all((i_m1*cs + i_m0) < m, (i_n1*cs + i_n0) < n)):
                                    output_args = i + [i_m1*cs + i_m0, i_n1*cs + i_n0]
                                    input_args = i + [i_n1, i_m1, i_m0, i_n0]
                                    ib.store(output, output_args,
                                             ib.load(input_, input_args))
            return ib.get()
        # If it is implemented with tvm.compute,
        # the generated stmt is difficult to process for poly in the fusion scene
        return tvm.extern(original_shape, [data], lambda ins, outs : kernel_ir(ins[0], outs[0]), name=output_name,
                          dtype=data.dtype)

    def _default2zn(data):
        shape = [get_const(x) for x in data.shape]
        dtype = data.dtype
        if len(shape) < 2:
            raise ValueError("length of shape of input_data should be greater than or equal to 2, but got %d"
                             % len(shape))
        m, n = shape[-2:]
        output_shape = []
        for i in range(0, len(shape) - 2):
            output_shape.append(shape[i])
        m1 = (m + cs - 1) // cs
        n1 = (n + cs - 1) // cs
        output_shape.extend([n1, m1, cs, cs])

        def fcompute(*output_indices):
            input_indices = []
            batch_len = len(output_indices) - 4
            n1_indice = output_indices[batch_len]
            m1_indice = output_indices[batch_len + 1]
            m0_indcie = output_indices[batch_len + 2]
            n0_indcie = output_indices[batch_len + 3]
            m_indice = m1_indice * cs + m0_indcie
            n_indice = n1_indice * cs + n0_indcie
            for i in range(0, batch_len):
                input_indices.append(output_indices[i])
            input_indices.append(m_indice)
            input_indices.append(n_indice)
            res = tvm.if_then_else(tvm.any(m_indice >= m, n_indice >= n), tvm.const(0, dtype), data(*input_indices))
            return res
        output = tvm.compute(output_shape, fcompute, name=output_name)
        return output

    # FRACTAL_NZ: zN fractal format
    if (src_format == "DefaultFormat" or src_format == "NCHW") and dst_format == "FRACTAL_NZ":
        return _default2zn(input_data)
    elif src_format == "FRACTAL_NZ" and (dst_format == "DefaultFormat" or dst_format == "NCHW"):
        if "output_shape" not in attrs:
            raise ValueError("output_shape(original_shape) not be found in the attrs")
        original_shape = attrs["output_shape"]
        return _zn2default(input_data, original_shape)
    else:
        raise ValueError("TransData for src_format %s and dst_format %s is not supported"
                         % (src_format, dst_format))

@tvm.register_func("Conv2D")
def conv2d_nhwc(inputs, attrs):
    attrs = {k: v for k, v in attrs.items()}
    # Check inputs and attrs
    if len(inputs) != 2:
        raise ValueError("length of inputs shoule be 2, but got %d." % len(inputs))
    if "stride" not in attrs:
        raise ValueError("stride not be found in the attrs")
    data = inputs[0]
    weight = inputs[1]
    output_name = "T_conv2d_nhwc_" + data.op.name + "_" + weight.op.name
    stride = attrs["stride"]
    data_dtype = data.dtype
    weight_dtype = weight.dtype
    # Check data type
    vc_util.ops_dtype_check(data_dtype, vc_util.DtypeForDavinci.FLOAT16)
    vc_util.ops_dtype_check(weight_dtype, vc_util.DtypeForDavinci.FLOAT16)
    # Check shape
    if len(data.shape) != 4 or len(weight.shape) != 4:
        raise ValueError("shape of data and weight should be 4-dim, but got %d and %d." % (len(data.shape),
            len(weight.shape)))
    # Compute output
    n, in_h, in_w, in_c = data.shape
    out_c, k_h, k_w, in_c = weight.shape
    _, _, s_h, s_w = stride
    o_h = (in_h - k_h) // s_h + 1
    o_w =(in_w - k_w) // s_w + 1
    rc = tvm.reduce_axis((0, in_c), name="rc")
    rh = tvm.reduce_axis((0, k_h), name="rh")
    rw = tvm.reduce_axis((0, k_w), name="rw")
    output = tvm.compute(
        (n, o_h, o_w, out_c),
        lambda n, h, w, o: tvm.sum(
            data[n, (h * s_h + rh), (w * s_w + rw), rc]
            * weight[o, rh, rw, rc],
            axis=[rc, rh, rw]),
        name=output_name
    )
    return output

@tvm.register_func("CumSum")
def cumsum(inputs, attrs):
    if len(inputs) != 1:
        raise ValueError("length of inputs shoule be 1, but got %d." % len(inputs))
    in_tensor = inputs[0]
    shape = in_tensor.shape
    attrs = {k: v for k, v in attrs.items()}
    axis = int(attrs["axis"][0]) if "axis" in attrs else 0
    exclusive = attrs["exclusive"].value if "exclusive" in attrs else False
    reverse = attrs["reverse"].value if "reverse" in attrs else False
    output_name = "T_cumsum_" + in_tensor.op.name
    def kernel_ir(data, dst):
        ib = tvm.ir_builder.create()
        # axes before cumm-axis
        with ib.for_range_n(shape[:axis], "i0") as i0:
            # axes after cumm-axis
            with ib.for_range_n(shape[axis+1:], "i1") as i1:
                idx_0 = i0 + [0] + i1 if not reverse else i0 + [shape[axis] - 1] + i1
                ib.store(dst, idx_0,  ib.load(data, idx_0) if not exclusive else tvm.const(0, data.dtype))
                # iterate the cumm-axis to do cumulated sum (start from 1)
                with ib.for_range(1, shape[axis], name="cum_idx") as m:
                    idx_pre = i0 + [m - 1] + i1 if not reverse else i0 + [shape[axis] - m] + i1
                    idx_cur = i0 +[m] + i1 if not reverse else i0 + [shape[axis] - 1 - m] + i1
                    ib.store(dst, idx_cur, ib.load(dst, idx_pre) + ib.load(data, idx_cur if not exclusive else idx_pre))
        return ib.get()
    return tvm.extern(shape, [in_tensor], lambda ins, outs : kernel_ir(ins[0], outs[0]), name=output_name,
                          dtype=in_tensor.dtype)

@tvm.register_func("CumProd")
def cumprod(inputs, attrs):
    if len(inputs) != 1:
        raise ValueError("length of inputs shoule be 1, but got %d." % len(inputs))
    in_tensor = inputs[0]
    shape = in_tensor.shape
    attrs = {k: v for k, v in attrs.items()}
    axis = int(attrs["axis"][0]) if "axis" in attrs else 0
    exclusive = attrs["exclusive"].value if "exclusive" in attrs else False
    reverse = attrs["reverse"].value if "reverse" in attrs else False
    output_name = "T_cumprod_" + in_tensor.op.name
    def kernel_ir(data, dst):
        ib = tvm.ir_builder.create()
        # axes before cumm-axis
        with ib.for_range_n(shape[:axis], "i0") as i0:
            # axes after cumm-axis
            with ib.for_range_n(shape[axis+1:], "i1") as i1:
                idx_0 = i0 + [0] + i1 if not reverse else i0 + [shape[axis] - 1] + i1
                ib.store(dst, idx_0, tvm.const(1, data.dtype) if exclusive else ib.load(data, idx_0))
                # iterate the cumm-axis to do cumulated production (start from 1)
                with ib.for_range(1, shape[axis], name="cum_idx") as m:
                    idx_pre = i0 + [m - 1] + i1 if not reverse else i0 + [shape[axis] - m] + i1
                    idx_cur = i0 +[m] + i1 if not reverse else i0 + [shape[axis] - 1 - m] + i1
                    ib.store(dst, idx_cur, ib.load(dst, idx_pre) * ib.load(data, idx_pre if exclusive else idx_cur))
        return ib.get()
    return tvm.extern(shape, [in_tensor], lambda ins, outs : kernel_ir(ins[0], outs[0]), name=output_name,
                          dtype=in_tensor.dtype)