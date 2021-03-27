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
        m1 = m // cs
        n1 = n // cs
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
