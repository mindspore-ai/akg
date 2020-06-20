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

"""div compute"""

import akg.tvm
import akg
from akg import backend as tvm_cce
from akg.backend import cce_params as param

output_name_suffix = [0]
ub_name_suffix = [0]


def compute_four2five(inputs, raw_shape_4d):
    """
    4D to 5D transform

    Args:
        inputs (tvm.tensor.Tensor): Input data.
        raw_shape_4d (Union[tuple, list]): he raw shape of tensor
            format: (in_n, in_c, in_h, in_w)

    Returns:
        Tensor
    """

    n, c, h, w = raw_shape_4d
    c1 = (c + 16 - 1) // 16
    output_name_suffix[0] += 1
    return akg.tvm.extern([(n, c1, h, w, 16)], [inputs],
                          lambda ins, outs: four2five_ir(ins[0], raw_shape_4d, outs[0]),
                          dtype=[inputs.dtype], name="output_" + hex(output_name_suffix[0]))


def compute_five2four(inputs, raw_shape_4d):
    """
    5D to 4D transform

    Args:
        inputs (tvm.tensor.Tensor): input data.
        raw_shape_4d (Union[tuple, list]): the raw shape of tensor.
            format: (in_n, in_c, in_h, in_w)

    Returns:
        Tensor
    """
    n, c, h, w = raw_shape_4d
    output_name_suffix[0] += 1
    return akg.tvm.extern([(n, c, h, w)], [inputs],
                          lambda ins, outs: five2four_ir(ins[0], raw_shape_4d, outs[0]),
                          dtype=[inputs.dtype], name="output_" + hex(output_name_suffix[0]))


def decl_memory(buffer_scope):
    """
    memory declaration

    Args:
        buffer_scope (str): the scope of ubuferr

    Returns:
        node.
    """
    @akg.tvm.register_func("tvm.info.mem.%s" % buffer_scope)
    def mem_info_ub_buffer():
        return akg.tvm.make.node("MemoryInfo",
                                 unit_bits=32 * 8,
                                 max_simd_bits=32 * 8,
                                 max_num_bits=tvm_cce.CceProductParams().get_params_("Unified_Buffer") * 8,
                                 head_address=akg.tvm.const(0, 'int32'))


def allocate_ub(ib, dtype, size, name):
    """
    UB allocation

    Args:
        ib: Instance of ir_builder.
        dtype (str): Support data type: float16.
        size (int): The size to allocate.
        name (str): The name of allocated ub. in_node suffix name must bu "local.UB"

    Returns:
        ub buffer
    """
    name = name + ".local.UB" + hex(ub_name_suffix[0])
    scope = "local.UB" + hex(ub_name_suffix[0])
    decl_memory(scope)
    buf_var = ib.allocate(dtype, (size,), name, scope=scope)
    return akg.tvm.decl_buffer((size,), dtype, name, scope=scope, data=buf_var)


def emit_copy_gm_ubuf(ib, cmd, dtype, size, dst, dst_offset, src, src_offset):
    """
    copy_gm_ubuf emit

    Args:
        ib: Instance of ir_builder.
        cmd (str): commond type, copy_gm_to_ubuf or copy_ubuf_to_gm.
        dtype (str): Support data type: float16.
        size (int): The size of copy data.
        dst: The addr copy to.
        dst_offset: The offset of dst addr.
        src: The addr copy form.
        src_offset: The offset of src addr.

    Returns:
        in_none
    """
    n_burst = size // (2 ** 16 * 32)
    tail = size % (2 ** 16 * 32)
    burst_ele_num = 256 // akg.get_bit_len(dtype)

    if tail > 0:
        dst_offset = dst_offset + n_burst * (2 ** 16 * 32)
        src_offset = src_offset + n_burst * (2 ** 16 * 32)
        len_burst = (tail + burst_ele_num - 1) // burst_ele_num
        ib.emit(akg.tvm.call_extern(
            dtype, cmd,
            dst.access_ptr("w", offset=dst_offset),
            src.access_ptr("rw", offset=src_offset),
            0, 1, len_burst, 0, 0))


def set_array_config(ib, addr_array, addr_array_buf, src_array_stride, dst_array_stride, output_offset):
    """
    set arrary configuration

    Args:
        ib: Instance of ir_builder.
        addr_array: The array of addr.
        addr_array_buf: The array of buf addr
        src_array_stride (int): src stride of scatter_vnchwconv_b16.
        dst_array_stride (int): dst stride of scatter_vnchwconv_b16.
        output_offset (int): The offset of output(ub).

    Returns:
        in_none
    """
    src0_offset = 8 * 0
    src1_offset = 8 * 1
    dst0_offset = 8 * 2
    dst1_offset = 8 * 3
    dtype_len = 2

    with ib.for_range(0, 8, name="i") as i:
        ib.emit(akg.tvm.call_extern("uint64", "reg_mov",
                                    akg.tvm.call_extern(addr_array.dtype, "reg", addr_array[src0_offset + i]),
                                    dtype_len * src_array_stride * i))

        ib.emit(akg.tvm.call_extern("uint64", "reg_mov",
                                    akg.tvm.call_extern(addr_array.dtype, "reg", addr_array[src1_offset + i]),
                                    dtype_len * src_array_stride * (i + 8)))

        ib.emit(akg.tvm.call_extern("uint64", "reg_mov",
                                    akg.tvm.call_extern(addr_array.dtype, "reg", addr_array[dst0_offset + i]),
                                    dtype_len * (output_offset + dst_array_stride * i)))

        ib.emit(akg.tvm.call_extern("uint64", "reg_mov",
                                    akg.tvm.call_extern(addr_array.dtype, "reg", addr_array[dst1_offset + i]),
                                    dtype_len * (output_offset + dst_array_stride * (i + 8))))

    ib.emit(akg.tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw", offset=src0_offset)
                                )
            )
    ib.emit(akg.tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw", offset=src1_offset)
                                )
            )
    ib.emit(akg.tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw", offset=dst0_offset)
                                )
            )
    ib.emit(akg.tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw", offset=dst1_offset)
                                )
            )


def get_vnchwconv_cube_buf_max(actual_col_size, is_four2five):
    """
    get max size of cube buffer

    Args:
        actual_col_size (int):  the size of actual colume (h*w)
        is_four2five (bool): operate type.

    Returns:
        int int.
        cube_size, tail_cube_size.
    """
    actual_cube_buf_size = ((actual_col_size - 1) // 16 + 1) * 16 * 16 * 2  # Byte
    ub_cut_upper_limit = (tvm_cce.CceProductParams().get_params_("Unified_Buffer")) // 2  # Byte
    if actual_cube_buf_size > ub_cut_upper_limit:
        if is_four2five:
            tail_cube_size = actual_cube_buf_size % ub_cut_upper_limit
            return (ub_cut_upper_limit // 2), (tail_cube_size // 2)  # fp16
        if actual_col_size % 16 != 0:
            # remain ubuf C0*C0*type_len(fp16) = 512 Byte
            ub_cut_upper_limit = (tvm_cce.CceProductParams().get_params_("Unified_Buffer") - 512) // 2  # Byte
            ub_cut_upper_limit = ub_cut_upper_limit // (2 * 16 * 16) * (2 * 16 * 16)  # Byte
        tail_cube_size = actual_cube_buf_size % ub_cut_upper_limit
        return (ub_cut_upper_limit // 2), (tail_cube_size // 2)  # fp16
    return (actual_cube_buf_size // 2), 0  # fp16


def four2five_ir(inputs, shape_4d, output):
    """
    4D to 5D IR

    Args:
        inputs (tvm.tensor.Tensor): Input tensor.
        shape_4d (list): The raw shape fo Tensor (in_n, in_c, in_h, in_w).
        output (tvm.tensor.Tensor): Output tensor.

    Returns:
        the result statement.
    """
    in_n, in_c, in_h, in_w = shape_4d
    c0c0 = 16
    actual_col_size = in_h * in_w
    vnchwconv_cube_buf_max, tail_cube_size = get_vnchwconv_cube_buf_max(actual_col_size, True)
    vnchwconv_cube_col_size = vnchwconv_cube_buf_max // c0c0
    ib = akg.tvm.ir_builder.create()
    ub_name_suffix[0] += 1
    input_ub = allocate_ub(ib, inputs.dtype, vnchwconv_cube_buf_max, "input_ub")
    output_ub = allocate_ub(ib, output.dtype, vnchwconv_cube_buf_max, "output_ub")
    addr_array = ib.allocate("uint64", (32,), name="addr_array", scope=param.scope_reg)
    addr_array_buf = akg.tvm.decl_buffer((32,), "uint64_t", "addr_array_buf", scope=param.scope_reg, data=addr_array)

    src_array_stride = vnchwconv_cube_col_size
    dst_array_stride = c0c0
    output_offset = vnchwconv_cube_buf_max
    set_array_config(ib, addr_array, addr_array_buf, src_array_stride, dst_array_stride, output_offset)

    c1_range = (in_c + c0c0 - 1) // c0c0
    buf_cube_range = (actual_col_size + vnchwconv_cube_col_size - 1) // vnchwconv_cube_col_size

    with ib.for_range(0, in_n, name="n") as n:
        with ib.for_range(0, c1_range, name="c1") as c1:
            with ib.for_range(0, buf_cube_range, name="buf_cube_index") as cube_i:
                if buf_cube_range > 1:
                    # Standard processing:
                    # in_copy gm to ub in 16 copies --> Update transpose config -->
                    # Transposed data --> in_copy ub to gm at one-time
                    with ib.if_scope(cube_i != buf_cube_range - 1):
                        repeat = vnchwconv_cube_col_size // 16
                        src_stride = 0 if repeat == 1 else 1
                        dst_stride = 0 if repeat == 1 else 16
                        with ib.for_range(0, c0c0, name="col") as col:
                            emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", inputs.dtype, vnchwconv_cube_col_size,
                                              input_ub, col * vnchwconv_cube_col_size,
                                              inputs, n * in_c * in_h * in_w + (c1 * c0c0 + col) * actual_col_size +
                                              cube_i * vnchwconv_cube_col_size)

                        ib.emit(akg.tvm.call_extern("int32",
                                                    "scatter_vnchwconv_b16",
                                                    "VA2",
                                                    "VA0",
                                                    repeat,
                                                    dst_stride,
                                                    src_stride
                                                    )
                                )

                        emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, vnchwconv_cube_buf_max,
                                          output, n * in_c * in_h * in_w + c1 * c0c0 * actual_col_size +
                                          cube_i * vnchwconv_cube_buf_max,
                                          output_ub, 0)
                    with ib.else_scope():
                        # Tail processing:
                        # in_copy gm to ub in 16 copies --> Update transpose config -->
                        # Transposed data --> in_copy ub to gm at one-time
                        repeat = tail_cube_size // 16 // 16
                        src_stride = 0 if repeat == 1 else 1
                        dst_stride = 0 if repeat == 1 else 16
                        with ib.for_range(0, c0c0, name="col") as col:
                            emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", inputs.dtype, tail_cube_size // 16,
                                              input_ub, col * vnchwconv_cube_col_size,
                                              inputs, n * in_c * in_h * in_w + (c1 * c0c0 + col) * actual_col_size +
                                              cube_i * vnchwconv_cube_col_size)

                        ib.emit(akg.tvm.call_extern("int32",
                                                    "scatter_vnchwconv_b16",
                                                    "VA2",
                                                    "VA0",
                                                    repeat,
                                                    dst_stride,
                                                    src_stride
                                                    )
                                )

                        emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype,
                                          (actual_col_size % vnchwconv_cube_col_size) * c0c0,
                                          output, n * in_c * in_h * in_w + c1 * c0c0 * actual_col_size +
                                          cube_i * vnchwconv_cube_buf_max,
                                          output_ub, 0)
                else:
                    # Standard processing:
                    # in_copy gm to ub in 16 copies --> Update transpose config -->
                    # Transposed data --> in_copy ub to gm at one-time
                    repeat = vnchwconv_cube_col_size // 16
                    src_stride = 0 if repeat == 1 else 1
                    dst_stride = 0 if repeat == 1 else 16

                    with ib.for_range(0, c0c0, name="col") as col:
                        emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", inputs.dtype, vnchwconv_cube_col_size,
                                          input_ub, col * vnchwconv_cube_col_size,
                                          inputs, n * in_c * in_h * in_w + (c1 * c0c0 + col) * actual_col_size +
                                          cube_i * vnchwconv_cube_col_size)

                    ib.emit(akg.tvm.call_extern("int32",
                                                "scatter_vnchwconv_b16",
                                                "VA2",
                                                "VA0",
                                                repeat,
                                                dst_stride,
                                                src_stride
                                                )
                            )

                    emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, actual_col_size * c0c0,
                                      output, n * in_c * in_h * in_w + c1 * c0c0 * actual_col_size +
                                      cube_i * vnchwconv_cube_buf_max,
                                      output_ub, 0)

    return ib.get()


def five2four_ir(inputs, shape_4d, output):
    """
    5D to 4D IR

    Args:
        inputs (tvm.tensor.Tensor): input tensor.
        shape_4d (list): The raw shape fo Tensor (in_n, in_c, in_h, in_w).
        output (tvm.tensor.Tensor): output tensor.

    Returns:
        the result statement.
    """
    in_n, in_c, in_h, in_w = shape_4d
    c0c0 = 16
    actual_col_size = in_h * in_w
    vnchwconv_cube_buf_max, tail_cube_size = get_vnchwconv_cube_buf_max(actual_col_size, False)
    vnchwconv_cube_col_size = vnchwconv_cube_buf_max // c0c0
    ib = akg.tvm.ir_builder.create()
    ub_name_suffix[0] += 1
    input_ub = allocate_ub(ib, inputs.dtype, vnchwconv_cube_buf_max, "input_ub")
    output_ub = allocate_ub(ib, output.dtype, vnchwconv_cube_buf_max, "output_ub")
    addr_array = ib.allocate("uint64", (32,), name="addr_array", scope=param.scope_reg)
    addr_array_buf = akg.tvm.decl_buffer((32,), "uint64_t", "addr_array_buf", scope=param.scope_reg, data=addr_array)

    src_array_stride = c0c0
    dst_array_stride = vnchwconv_cube_col_size
    output_offset = vnchwconv_cube_buf_max
    set_array_config(ib, addr_array, addr_array_buf, src_array_stride, dst_array_stride, output_offset)

    c1_range = (in_c + c0c0 - 1) // c0c0
    buf_cube_range = (actual_col_size + vnchwconv_cube_col_size - 1) // vnchwconv_cube_col_size
    hw_tail_size = actual_col_size % vnchwconv_cube_col_size

    if hw_tail_size != 0 and hw_tail_size % c0c0 != 0 and buf_cube_range > 1:
        output_head_ub = allocate_ub(ib, output.dtype, c0c0 * c0c0, "output_head_ub")

    with ib.for_range(0, in_n, name="n") as n:
        with ib.for_range(0, c1_range, name="c1") as c1:
            with ib.for_range(0, buf_cube_range, name="buf_cube_index") as cube_i:
                if buf_cube_range > 1:
                    with ib.if_scope(cube_i != buf_cube_range - 1):
                        # Standard processing:
                        # in_copy gm to ub in one-time --> Update transpose config -->
                        # Transposed data --> in_copy ub to gm in 16 copies
                        repeat = vnchwconv_cube_col_size // 16
                        src_stride = 0 if repeat == 1 else 16
                        dst_stride = 0 if repeat == 1 else 1

                        emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", inputs.dtype, vnchwconv_cube_buf_max,
                                          input_ub, 0,
                                          inputs, n * c1_range * c0c0 * in_h * in_w + c1 * c0c0 * actual_col_size +
                                          cube_i * vnchwconv_cube_buf_max)

                        ib.emit(akg.tvm.call_extern("int32",
                                                    "scatter_vnchwconv_b16",
                                                    "VA2",
                                                    "VA0",
                                                    repeat,
                                                    dst_stride,
                                                    src_stride
                                                    )
                                )

                        with ib.for_range(0, c0c0, name="col") as col:
                            emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, vnchwconv_cube_col_size,
                                              output,
                                              n * in_c * in_h * in_w +
                                              c1 * c0c0 * actual_col_size +
                                              col * actual_col_size,
                                              output_ub, col * vnchwconv_cube_col_size)

                    with ib.else_scope():
                        # Tail processing:
                        # in_copy gm to ub in one-time --> Update transpose config -->
                        # Transposed data --> in_copy ub to gm in 16 copies
                        repeat = tail_cube_size // 16 // 16
                        src_stride = 0 if repeat == 1 else 16
                        dst_stride = 0 if repeat == 1 else 1

                        emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", inputs.dtype, tail_cube_size,
                                          input_ub, 0,
                                          inputs, n * c1_range * c0c0 * in_h * in_w + c1 * c0c0 * actual_col_size +
                                          cube_i * vnchwconv_cube_buf_max)

                        ib.emit(akg.tvm.call_extern("int32",
                                                    "scatter_vnchwconv_b16",
                                                    "VA2",
                                                    "VA0",
                                                    repeat,
                                                    dst_stride,
                                                    src_stride
                                                    )
                                )

                        if hw_tail_size % c0c0 == 0:
                            with ib.for_range(0, c0c0, name="col") as col:
                                emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, hw_tail_size,
                                                  output,
                                                  n * in_c * in_h * in_w +
                                                  c1 * c0c0 * actual_col_size +
                                                  col * actual_col_size +
                                                  cube_i * vnchwconv_cube_col_size,
                                                  output_ub, col * vnchwconv_cube_col_size)
                        else:
                            # Exception processing: the actual colume size cann't be divided by 16
                            # 1. in_copy head data (16*16) from gm for backup
                            # 2. in_copy tail data from ub to gm;(because of padding data rewrite the head)
                            # 3. in_copy head data (Step 1) to restore
                            with ib.for_range(0, c0c0, name="head_col_write") as head_col_write:
                                emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", output.dtype, c0c0,
                                                  output_head_ub, head_col_write * c0c0,
                                                  output,
                                                  n * in_c * in_h * in_w + c1 * c0c0 * actual_col_size +
                                                  head_col_write * actual_col_size)

                            with ib.for_range(0, c0c0, name="col") as col:
                                emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, hw_tail_size,
                                                  output,
                                                  n * in_c * in_h * in_w +
                                                  c1 * c0c0 * actual_col_size +
                                                  col * actual_col_size +
                                                  cube_i * vnchwconv_cube_col_size,
                                                  output_ub, col * vnchwconv_cube_col_size)

                            with ib.for_range(0, c0c0, name="head_col_read") as head_col_read:
                                emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, c0c0,
                                                  output,
                                                  n * in_c * in_h * in_w + c1 * c0c0 * actual_col_size +
                                                  head_col_read * actual_col_size,
                                                  output_head_ub, head_col_read * c0c0)
                else:
                    # Standard processing:
                    # in_copy gm to ub in one-time --> Update transpose config -->
                    # Transposed data --> in_copy ub to gm in 16 copies
                    repeat = vnchwconv_cube_col_size // 16
                    src_stride = 0 if repeat == 1 else 16
                    dst_stride = 0 if repeat == 1 else 1

                    emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", inputs.dtype, vnchwconv_cube_buf_max,
                                      input_ub, 0,
                                      inputs, n * c1_range * c0c0 * in_h * in_w + c1 * c0c0 * actual_col_size +
                                      cube_i * vnchwconv_cube_buf_max)

                    ib.emit(akg.tvm.call_extern("int32",
                                                "scatter_vnchwconv_b16",
                                                "VA2",
                                                "VA0",
                                                repeat,
                                                dst_stride,
                                                src_stride
                                                )
                            )

                    if hw_tail_size == 0:
                        with ib.for_range(0, c0c0, name="col") as col:
                            emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, vnchwconv_cube_col_size,
                                              output,
                                              n * in_c * in_h * in_w +
                                              c1 * c0c0 * actual_col_size +
                                              col * actual_col_size,
                                              output_ub, col * vnchwconv_cube_col_size)
                    else:
                        with ib.for_range(0, c0c0, name="col") as col:
                            with ib.new_scope():
                                ib.scope_attr(param.in_cin_cE_AXIS, "coproc_scope", 6)
                                emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, hw_tail_size,
                                                  output,
                                                  n * in_c * in_h * in_w +
                                                  c1 * c0c0 * actual_col_size +
                                                  col * actual_col_size,
                                                  output_ub, col * vnchwconv_cube_col_size)
    return ib.get()
