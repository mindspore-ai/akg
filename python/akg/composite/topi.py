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
from akg.tvm.hybrid import script
from akg.utils.format_transform import get_const
from akg.utils import validation_check as vc_util

import logging
import os
from pathlib import Path
import importlib.util


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

@tvm.register_func("StridedSlice")
def StridedSlice(inputs, attrs):
    in_tensor = inputs[0]
    shape = in_tensor.shape
    begin = list(attrs["begin"])
    end = list(attrs["end"])
    strides = list(attrs["strides"])
    slice_len = len(begin)

    begin_pos = [0] if "begin_mask" not in attrs else bin(int(attrs["begin_mask"]))[-1:1:-1]
    end_pos = [0] if "end_mask" not in attrs else bin(int(attrs["end_mask"]))[-1:1:-1]
    ellipsis_pos = [0] if "ellipsis_mask" not in attrs else bin(int(attrs["ellipsis_mask"]))[-1:1:-1]
    new_axis_pos = [0] if "new_axis_mask" not in attrs else bin(int(attrs["new_axis_mask"]))[-1:1:-1]
    shrink_axis_pos = [0] if "shrink_axis_mask" not in attrs else bin(int(attrs["shrink_axis_mask"]))[-1:1:-1]

    out_shape = []
    i, j = 0, 0
    has_ellipsis = False
    shrink_axes = []
    while i < slice_len or j < len(shape):
        if j >= slice_len or i >= slice_len:
            out_shape.append(shape[j])
            begin.append(0)
            end.append(shape[j])
            strides.append(1)
            i += 1
            j += 1
            continue
        if i < len(begin_pos) and begin_pos[i] == '1':
            begin[i] = -1 if int(strides[i]) < 0 else 0
        if i < len(end_pos) and end_pos[i] == '1':
            end[i] = -1 if int(strides[i]) < 0 else shape[j]
        if i < len(ellipsis_pos) and ellipsis_pos[i] == '1':
            out_shape.append(shape[j])
            begin[i] = 0
            end[i] = shape[j]
            strides[i] = 1
            i += 1
            j += 1
            continue
        if i < len(new_axis_pos) and new_axis_pos[i] == '1':
            out_shape.append(1)
            begin[i] = 1
            end[i] = 1
            strides[i] = 1
            in_tensor = topi.expand_dims(in_tensor, i, 1)
            i += 1
            continue
        if i < len(shrink_axis_pos) and shrink_axis_pos[i] == '1':
            shrink_axes.append(i)
            i += 1
            j += 1
            continue
        if int(begin[i]) < 0:
            begin[i] += shape[j]
        if int(end[i]) < 0:
            end[i] += shape[j]
        if int(begin[i]) < 0:
            begin[i] = 0
        elif int(begin[i]) >= int(shape[j]):
            begin[i] = shape[j] - 1
        if int(end[i]) < 0:
            end[i] = -1
        elif int(end[i]) >= int(shape[j]):
            end[i] = shape[j]
        out_idx = (end[i] - begin[i]) // strides[i]
        if not int(out_idx * strides[i]) == int(end[i] - begin[i]):
            out_idx += 1
        out_shape.append(out_idx)
        i += 1
        j += 1
    def get_old_indices(indices, idx):
        old_indices = list(indices)
        old_indices.insert(idx, begin[idx])
        return old_indices
    for shrink_axis in reversed(shrink_axes):
        new_shape = list(in_tensor.shape)
        new_shape.pop(shrink_axis)
        if not new_shape:
            return tvm.compute([1], lambda *i: in_tensor(*i))
        in_tensor = tvm.compute(new_shape, lambda *indices: in_tensor(*get_old_indices(indices, shrink_axis)))
        begin.pop(shrink_axis)
        strides.pop(shrink_axis)
    return tvm.compute(out_shape, lambda *i: in_tensor(*[b + idx * s for b, s, idx in zip(begin, strides, i)]))

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

@tvm.register_func("UserDefined")
def user_defined(inputs, attrs):

    op_desc_attr = []
    source_str = ""
    op_imply_path = ""
    func_name = ""
    func_type = ""

    for ext_arg in attrs.items():
        attr_name = ext_arg[0]
        if attr_name == "func_source_str":
            source_str = ext_arg[1].value
        elif attr_name == "op_imply_path":
            op_imply_path = ext_arg[1].value
        elif attr_name == "func_name":
            func_name = ext_arg[1].value
        elif attr_name == "func_type":
            func_type = ext_arg[1].value
        elif not (attr_name == "akg" or "_format" in attr_name):
            # store the rest of op attr for op build
            op_desc_attr.append(ext_arg)

    op_attrs = []
    ir_builder_attrs = {}
    for ext_arg in op_desc_attr:
        if func_type == "ir_builder":
            # ir_builder functions take attrs as a dict/tvm.Map
            ir_builder_attrs[ext_arg[0]] = ext_arg[1]
        else:
            op_attrs.append(ext_arg[1])

    func_kernel = None
    if len(source_str) > 0:
        capture = locals()
        capture["source_str"] = source_str

        func_mod = compile(source_str, "", "exec")
        exec(func_mod)
        func_kernel = locals()[func_name]

    elif len(op_imply_path) > 0:
        if os.path.isfile(op_imply_path):
            custom_mod_name = Path(op_imply_path).resolve().stem
            mod_spec = importlib.util.spec_from_file_location(
                custom_mod_name, op_imply_path)
            custom_mod = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(custom_mod)
            func_kernel = getattr(custom_mod, func_name, None)
        else:
            logging.error("Can't find file under path: %s",
                          str(op_imply_path))
    else:
        logging.error(
            "Neither source_str nor op_imply_path is provided in the json file")

    if func_kernel is None:
        logging.error(
            "Failed in compiling op function from userdefine op")

    output = None
    if func_type == "hybrid":
        hybrid_func = script(func_kernel, capture=capture)
        inputs = list(inputs) + op_attrs
        output = hybrid_func(*inputs)
    elif func_type == "ir_builder":
        output = func_kernel(inputs, ir_builder_attrs)
    else:
        inputs = list(inputs) + op_attrs
        output = func_kernel(*inputs)

    return output

@tvm.register_func("GatherNd")
def gather_nd(inputs, attrs):
    if len(inputs) != 2:
        raise ValueError(f"2 inputs expected, but got {len(input)}")
    data, indices = inputs

    data_shape = list(data.shape)
    indices_shape = list(indices.shape)
    indices_last_dim = len(indices_shape) - 1
    left_shape = indices_shape[:indices_last_dim]
    right_shape = data_shape[int(indices_shape[indices_last_dim]):]
    def gen_ir(data, indices, out):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(left_shape, 'i') as i:
            with ib.for_range_n(right_shape, 'j') as j:
                read_idx = []
                inbound = True
                for k in range(0, int(indices_shape[-1])):
                    temp_idx = ib.load(indices, i + [k])
                    if k == 0:
                        inbound = tvm.all((temp_idx >= 0), (temp_idx < data_shape[k]))
                    else:
                        inbound = tvm.all(inbound, (temp_idx >= 0), (temp_idx < data_shape[k]))
                    read_idx.append(temp_idx)
                with ib.if_scope(inbound):
                    ib.store(out, i + j, ib.load(data, read_idx + j))
                with ib.else_scope():
                    ib.store(out, i + j, tvm.const(0, data.dtype))
        return ib.get()

    output_name = "T_gathernd_" + data.op.name + "_" + indices.op.name
    output_shape = left_shape + right_shape
    out_buf = tvm.decl_buffer(output_shape, data.dtype, output_name)
    return tvm.extern([output_shape], [data, indices], lambda ins, outs: gen_ir(ins[0], ins[1], outs[0]),
                                          dtype=data.dtype, out_buffers=[out_buf], name=output_name)

@tvm.register_func("TensorScatterAdd")
def tensor_scatter_add(inputs, attrs):
    if len(inputs) != 3:
        raise ValueError(f"3 inputs expected, but got {len(input)}")
    data, indices, updates = inputs
    data_shape = list(data.shape)
    indices_shape = list(indices.shape)
    is_1d_indices = False
    if len(indices_shape) == 1:
        indices_shape.append(1)
        is_1d_indices = True
    left_shape = indices_shape[:-1]
    right_shape = data_shape[int(indices_shape[-1]):]
    def gen_ir(data, indices, updates, out):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(left_shape, "i") as i:
            with ib.for_range_n(right_shape, "j") as j:
                index_read = i + j
                index_write = []
                inbound = True
                if is_1d_indices:
                    temp_idx = ib.load(indices, i)
                    inbound = tvm.all((temp_idx >= 0), (temp_idx < data_shape[0]))
                    index_write.append(temp_idx)
                else:
                    for k in range(0, int(indices_shape[-1])):
                        temp_idx = ib.load(indices, i+[k])
                        if k == 0:
                            inbound = tvm.all((temp_idx >= 0), (temp_idx < data_shape[k]))
                        else:
                            inbound = tvm.all(inbound, (temp_idx >= 0), (temp_idx < data_shape[k]))
                        index_write.append(temp_idx)
                index_write = index_write + j
                with ib.if_scope(inbound):
                    temp = ib.load(updates, index_read) + ib.load(out, index_write)
                    ib.store(out, index_write, temp)
        return ib.get()

    output_name = "T_tsa_" + data.op.name + "_" + indices.op.name + "_" + updates.op.name
    out_buf = tvm.decl_buffer(data.shape, data.dtype, output_name)
    return tvm.extern([data.shape], [data, indices, updates], lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
                                          dtype=data.dtype, out_buffers=[out_buf], name=output_name)

@tvm.register_func("UnsortedSegmentSum")
def tensor_unsorted_segment_sum(inputs, attrs):
    attrs = {k: v for k, v in attrs.items()}
    num = attrs['num_segments']
    op_id = attrs['op_id'] if 'op_id' in attrs else 0
    if len(inputs) != 2:
        raise ValueError(f"2 inputs expected, but got {len(input)}")
    data, indices = inputs
    data_shape = list(data.shape)
    indices_shape = list(indices.shape)
    segment_len = len(data_shape) - len(indices_shape)
    if segment_len < 0:
        raise ValueError(f'input rank should not be less than segment_id rank')
    for i, v in enumerate(indices_shape):
        if int(v) != int(data_shape[i]):
            raise ValueError(f'input shape at dim {i} is not equal to segment_id shape at dim {i}')
    output_shape = [num]
    if segment_len > 0:
        output_shape += data_shape[len(indices_shape):]
    if len(indices_shape) > 1:
        raise ValueError('only 1-D segment currently supported')

    def gen_ir(data, indices, out):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(indices_shape, "i") as i:
            read_idx = ib.load(indices, i)
            # 1-D segment
            with ib.for_range_n(data_shape[1:], 'j') as j:
                inbound = tvm.all((read_idx >= 0), (read_idx < num))
                with ib.if_scope(inbound):
                    val = ib.load(data, i + j) + ib.load(out, [read_idx] + j)
                    ib.store(out, [read_idx] + j, val)
        return ib.get()

    output_name = "T_uss_" + data.op.name + "_" + indices.op.name
    out_buf = tvm.decl_buffer(output_shape, data.dtype, output_name)
    return tvm.extern([data.shape], [data, indices], lambda ins, outs: gen_ir(ins[0], ins[1], outs[0]),
                                          dtype=data.dtype, out_buffers=[out_buf], name=output_name)

@tvm.register_func("Gather")
def gather(inputs, attrs):
    attrs = {k: v for k, v in attrs.items()}
    axis = int(attrs["axis"][0]) if "axis" in attrs else 0
    if len(inputs) != 2:
        raise ValueError(f"2 inputs expected, but got {len(input)}")
    data, indices = inputs
    data_shape = list(data.shape)
    indices_shape = list(indices.shape)
    output_shape = data_shape[: axis] + indices_shape + data_shape[axis + 1:]

    def gen_ir(data, indices, out):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(data_shape[: axis], "i") as i:
            with ib.for_range_n(indices_shape, "j") as j:
                load_idx = ib.load(indices, j)
                inbound = tvm.all(load_idx >= 0, load_idx < data_shape[axis])
                read_idx = i + [load_idx]
                with ib.for_range_n(data_shape[axis + 1:], "k") as k:
                    with ib.if_scope(inbound):
                        ib.store(out, i + j + k, ib.load(data, read_idx + k))
                    with ib.else_scope():
                        ib.store(out, i + j + k, tvm.const(0, data.dtype))
        return ib.get()

    output_name = "T_gather_" + data.op.name + "_" + indices.op.name + "_" + str(axis)
    out_buf = tvm.decl_buffer(output_shape, data.dtype, output_name)
    return tvm.extern([data.shape], [data, indices], lambda ins, outs: gen_ir(ins[0], ins[1], outs[0]),
                                          dtype=data.dtype, out_buffers=[out_buf], name=output_name)


@tvm.register_func("StandardNormal")
def standard_normal(inputs, attrs):
    attrs = {k: v for k, v in attrs.items()}
    seed = attrs["seed"]
    shape = attrs["shape"]
    dtype = "float32"
    def gen_ir(out):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(shape, "i") as i:
            temp = ib.extern_call(seed, op_name="StandardNormal", dtype=dtype)
            ib.store(out, i, temp)
        return ib.get()

    output_name = "randnorm"
    out_buf = tvm.decl_buffer(shape, dtype, "res")
    return tvm.extern([shape],
                      [],
                      lambda ins, outs: gen_ir(outs[0]),
                      dtype=dtype,
                      out_buffers=[out_buf],
                      name=output_name)
